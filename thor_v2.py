import numpy as np
# from credit_docs.questnet_brc_bri import *
from credit_docs.api_bank_statements_new import *
# from credit_docs.wz_bank_statements import *
# from credit_docs.wz_cbs import *
from helpers import config, database_service
from credit_docs.noa import *
import pandas as pd
from numpy import NINF, inf
from apiclient import discovery
import re
from google.oauth2 import service_account
import boto3

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import statsmodels.api as sm
from statsmodels.nonparametric.api import KernelReg
import pickle
from googleapiclient.discovery import build
import snowflake.connector
import logging
import sys

logger = logging.getLogger()
handler = logging.StreamHandler()
logger.addHandler(handler)
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.setLevel('INFO')
from utils.error import ModelError, ErrorLevel

mod_err = ModelError()

"""
Changes made according to https://fundingasiagroup.atlassian.net/wiki/spaces/CRED/pages/846463161/ESG+-+EFS+Products
Grant financing to go through normal bolt model (i.e. 6 months P+I) , no more bullet repayment.
No more hard reject test bucket -- so we can remove it from the model.
Change in product treatment & limits
| - Maximum Quantum : $100K
| - Maximum Tenor : 12 months
| - No appetite for high risk industries
| - Quantum limits & tenor to be in accordance to industry's risk grading (refer to the confluence page)
Change in industries risk grading (refer to the confluence page)
"""
region_name = "ap-southeast-1"
runtime_client = boto3.client(service_name='sagemaker-runtime', region_name=region_name)
# prod_cfg = config.get_config()['prod_db']
productionDBLoan = database_service.DataService(
    host=prod_cfg['host'],
    username=prod_cfg['username'],
    password=prod_cfg['password'],
    port=prod_cfg['port'],
    database='loan_db')

productionDBMember = database_service.DataService(
    host=prod_cfg['host'],
    username=prod_cfg['username'],
    password=prod_cfg['password'],
    port=prod_cfg['port'],
    database='member_db')

datalake_cfg = config.get_config()['datalake_write']
datalakeDB = database_service.DataService(
    host=datalake_cfg['host'],
    username=datalake_cfg['username'],
    password=datalake_cfg['password'],
    port=datalake_cfg['port'],
    database='datalake')

snowflake_cfg = config.get_config()['snowflake']

rds_cfg = config.get_config()['data_rds_prod']
productionRDSResults = database_service.DataService(
    host=rds_cfg['host'],
    username=rds_cfg['username'],
    password=rds_cfg['password'],
    port=rds_cfg['port'],
    database=rds_cfg['database'])


class thor:

    def __init__(self, industryRiskMap, riskQuantumTenorMap, othParams, loanID):

        self.loanID = loanID
        self.applicationDate = self.get_application_date()
        self.applicationYrMth = self.get_application_yrmth()
        self.loanQT = self.get_loan_application_quantum_tenor()
        self.borrowerID = self.get_borrower_id()
        self.historicalDPD = self.get_historical_dpd()
        self.industryRiskMap = industryRiskMap
        self.riskQuantumTenorMap = riskQuantumTenorMap
        self.othParams = othParams

        # NOTE TO DEV: Pull from GS --done
        try:
            self.bankStatements, self.ctos_input, self.industry_sector = self.fetch_model_inputs()
            print("self.bankStatements\n", self.bankStatements.to_string())
            print("self.ctos_input\n", self.ctos_input.to_string())
            print("self.industry_sector\n", self.industry_sector)
            # print("self.bankStatements\n", self.bankStatements.dtypes)
            # print("self.ctos\n", self.ctos_input.dtypes)
        except Exception as e:
            mod_err.raise_exception(str(e), ErrorLevel.WARNING,
                                    data=f'Failed to get the Bank statements, Ctos and industry sector data for loan ID: {self.loanID}')
            self.bankStatements = pd.DataFrame()
            self.bankStatements['bankEndBalances'] = np.NaN
            self.bankStatements['bankLiabilities'] = np.NaN
            self.bankStatements['bankCredit'] = np.NaN
            self.bankStatements['bankDebit'] = np.NaN
            self.bankStatements['bankEndBalancesPct'] = np.NaN
            self.bankStatements['bankLiabilitiesPct'] = np.NaN
            self.bankStatements['bankCreditPct'] = np.NaN
            self.bankStatements['bankDebitPct'] = np.NaN
            self.bankStatements['DCRatio'] = np.NaN
            self.bankStatements['balanceCreditRatio'] = np.NaN
            self.bankStatements['balanceDebitRatio'] = np.NaN
            self.bankStatements['balanceLiabilityRatio'] = np.NaN
            self.bankStatements['DCRatioPct'] = np.NaN
            self.bankStatements['balanceCreditRatioPct'] = np.NaN
            self.bankStatements['balanceDebitRatioPct'] = np.NaN
            self.bankStatements['balanceLiabilityRatioPct'] = np.NaN
            self.bankStatements['bankStatementMonths'] = 0
            self.ctos_input = pd.DataFrame()
            self.industry_sector = np.NaN

        self.disbursedLoanCount = self.get_historical_disbursed_loan_count()
        # self.companyDetails = self.get_company_details()
        # self.companyDetails = parse_brc_company_details(self.brcReportID)
        # self.get_company_risk()

        self.riskQuantum = None
        self.riskTenor = None
        # self.get_internal_caps()  # Mark: Added as part of Nov 20 changes
        # self.industryBlacklistFlag = self.get_industry_blacklist_flag()

        ###
        self.applicationData = self.invoke()
        ###
        # self.industryCheck = None
        # self.hardRejectCheck = None
        self.quantumCap = None
        self.tenorCap = None
        self.modelInput = None

        self.min_quantum = 3000  # Minimum quantum for bolt product
        # Note: Not hardcoded, fetching them from gsheet. It is just initialization.
        self.ead = .60
        self.lgd = 1
        self.int_floor = None
        self.int_limit = None
        self.ead = self.get_oth_params('EAD')
        self.lgd = self.get_oth_params('LGD')
        self.int_floor = self.get_oth_params('int_floor')
        self.int_limit = self.get_oth_params('int_limit')

        logger.info(f"ead {self.ead}")
        logger.info(f"lgd {self.lgd}")

    def google_sheet_to_dataframe(self, sheet_id, sheet_name, sheet_range=''):
        scope = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/spreadsheets']
        credentials = service_account.Credentials.from_service_account_file(
            'client_secret.json', scopes=scope)
        SERVICE_NAME = 'sheets'
        API_VERSION = 'v4'
        SERVICE = discovery.build(SERVICE_NAME, API_VERSION, credentials=credentials)
        read_range = ("!".join([sheet_name, sheet_range]))
        result = SERVICE.spreadsheets().values().get(spreadsheetId=sheet_id, range=read_range).execute()
        rows = result['values']
        column_names = rows[0]
        m = re.compile('[^\w\d]+')
        column_names = [re.sub(m, '_', i).strip().upper() for i in column_names]
        df = pd.DataFrame(rows[1:], columns=column_names)
        return df[df['LOAN_ID'] == str(self.loanID)]

    def fetch_model_inputs(self):
        sheet_id = '1ohf_211FR14imGScxEowzsLNJ0ETlMz5PIt2Sta7MPA'
        sheet_name = 'Model Port (MY BOLT)'
        sheet_range = 'B3:AQ2000'
        df = self.google_sheet_to_dataframe(sheet_id, sheet_name, sheet_range)
        # print(df.to_string())
        # Separating both the Inputs
        df_bank = df.loc[:, "AVGBANKMONTHENDBALANCES":]
        df_ctos = df.loc[:, :"TOTALRECOURSECCRISAPPLICATIONCOUNT"]
        industry_sector = df.loc[:, "INDUSTRYSECTOR"].values[0]
        # print(df_bank.to_string())
        # print("\n\n\n\n\n")
        # print(df_ctos.to_string())
        return self.transform_bs(df_bank), self.transform_ctos(df_ctos), industry_sector

    @staticmethod
    def transform_bs(df_bank):
        df_bank_transform = pd.DataFrame(columns=["monthly_debit_pct", "dc_ratio_pct", "monthly_liabilities",
                                                  "bal_credit_ratio_pct", "bal_liab_ratio", "monthly_credit_pct",
                                                  "monthly_end_balance_pct", "bal_debit_ratio_pct", 'monthly_credit'])
        df_bank_transform["monthly_debit_pct"] = df_bank["BANKDEBITPCT"]
        df_bank_transform["dc_ratio_pct"] = df_bank["DCRATIOPCT"]
        df_bank_transform["monthly_liabilities"] = df_bank["BANKLIABILITIES"]
        df_bank_transform["bal_credit_ratio_pct"] = df_bank["BALANCECREDITRATIOPCT"]
        df_bank_transform["bal_liab_ratio"] = df_bank["BALANCELIABILITYRATIO"]
        df_bank_transform["monthly_credit_pct"] = df_bank["BANKCREDITPCT"]
        df_bank_transform["monthly_end_balance_pct"] = df_bank["BANKENDBALANCESPCT"]
        df_bank_transform["bal_debit_ratio_pct"] = df_bank["BALANCEDEBITRATIOPCT"]
        df_bank_transform["monthly_credit"] = df_bank["BANKCREDIT"]

        df_bank_transform = df_bank_transform.replace(np.inf, 333333333333)
        df_bank_transform = df_bank_transform.replace(-np.inf, -444444444444)
        df_bank_transform = df_bank_transform.replace('#N/A', np.nan)
        df_bank_transform = df_bank_transform.replace('None', np.nan)
        df_bank_transform = df_bank_transform.fillna(999999999999)
        return df_bank_transform

    @staticmethod
    def transform_ctos(df_ctos):
        df_ctos_transform = pd.DataFrame(columns=["onehotencoder__x1_ACTIVE", "onehotencoder__x0_nan",
                                                  "ccris_facility_arrears",
                                                  "ccris_application_total", "ccris_facility_total", "recourse_count",
                                                  "facility_value_normalised",
                                                  "total_recourse_ccris_application_count_approved",
                                                  "total_recourse_ccris_facility_count_arrears",
                                                  "ccris_application_approved", "onehotencoder__x0_PRIVATE LIMITED",
                                                  "onehotencoder__x1_EXISTING", "ccris_facility_value",
                                                  "recourse_ccris_application_count_total", "onehotencoder__x1_EXPIRED",
                                                  "onehotencoder__x1_nan", "cpy_years_of_establishment",
                                                  "onehotencoder__x0_SP", "onehotencoder__x0_PN"])
        # fill-1
        company_status = {
            'ACTIVE': [1, 0, 0, 0],
            'EXISTING': [0, 1, 0, 0],
            'EXPIRED': [0, 0, 1, 0],
            'nan': [0, 0, 0, 1]
        }
        company_type = {
            'nan': [1, 0, 0, 0],
            'PRIVATE LIMITED': [0, 1, 0, 0],
            'SP': [0, 0, 1, 0],
            'PN': [0, 0, 0, 1]
        }
        # print("\n\n\n", company_status.get(df_ctos["X"][self.row_number - 4], [0, 0, 0, 1])[0], "\n\n\n")
        df_ctos_transform.loc[0, "onehotencoder__x1_ACTIVE"] = \
            company_status.get(df_ctos["COMPANYSTATUS"].values[0].upper(), [0, 0, 0, 1])[0]
        df_ctos_transform["onehotencoder__x0_nan"] = company_type.get(df_ctos["COMPANYTYPE"].values[0], [1, 0, 0, 0])[0]

        df_ctos_transform["ccris_facility_arrears"] = df_ctos["CCRISFACILITYARREARS_1_0_"]  #
        df_ctos_transform["ccris_application_total"] = df_ctos["CCRISAPPLICATIONTOTALCOUNT"]  #
        df_ctos_transform["ccris_facility_total"] = df_ctos["CCRISOSTDFACILITYCOUNT"]  #
        df_ctos_transform["recourse_count"] = df_ctos["RECOURSECOUNT"]  #
        df_ctos_transform["facility_value_normalised"] = np.double(df_ctos['CCRISOSTDFACILITYVALUE']) / np.double(
            df_ctos['CCRISOSTDFACILITYCOUNT'])

        df_ctos_transform["total_recourse_ccris_application_count_approved"] = df_ctos[
            "TOTALRECOURSECCRISAPPLICATIONAPPROVEDCOUNT"]  #
        df_ctos_transform["total_recourse_ccris_facility_count_arrears"] = df_ctos[
            "TOTALRECOURSECOUNTCCRISFACILITYARREARS"]  #
        df_ctos_transform["ccris_application_approved"] = df_ctos["CCRISAPPLICATIONAPPROVEDCOUNT"]  #

        # fill-2
        df_ctos_transform["onehotencoder__x0_PRIVATE LIMITED"] = \
            company_type.get(df_ctos["COMPANYTYPE"].values[0], [1, 0, 0, 0])[1]
        df_ctos_transform["onehotencoder__x1_EXISTING"] = \
            company_status.get(df_ctos["COMPANYSTATUS"].values[0].upper(), [0, 0, 0, 1])[1]

        df_ctos_transform["ccris_facility_value"] = df_ctos["CCRISOSTDFACILITYVALUE"]  #
        df_ctos_transform["recourse_ccris_application_count_total"] = df_ctos["TOTALRECOURSECCRISAPPLICATIONCOUNT"]  #

        # fill-3
        df_ctos_transform["onehotencoder__x1_EXPIRED"] = \
            company_status.get(df_ctos["COMPANYSTATUS"].values[0].upper(), [0, 0, 0, 1])[2]
        df_ctos_transform["onehotencoder__x1_nan"] = \
            company_status.get(df_ctos["COMPANYSTATUS"].values[0].upper(), [0, 0, 0, 1])[3]

        df_ctos_transform["cpy_years_of_establishment"] = df_ctos["YEARSOFESTABLISHMENT"]  #

        # fill-4
        df_ctos_transform["onehotencoder__x0_SP"] = \
            company_type.get(df_ctos["COMPANYTYPE"].values[0], [1, 0, 0, 0])[2]
        df_ctos_transform["onehotencoder__x0_PN"] = \
            company_type.get(df_ctos["COMPANYTYPE"].values[0], [1, 0, 0, 0])[3]

        df_ctos_transform = df_ctos_transform.replace(np.inf, 333333333333)
        df_ctos_transform = df_ctos_transform.replace(-np.inf, -444444444444)
        df_ctos_transform = df_ctos_transform.replace('#N/A', np.nan)
        df_ctos_transform = df_ctos_transform.replace('None', np.nan)
        df_ctos_transform = df_ctos_transform.fillna(999999999999)
        return df_ctos_transform

    @staticmethod
    def predict_bs(feature_list_bs):
        content_type = "application/json"
        request_body = {"Input": [feature_list_bs]}
        data = json.loads(json.dumps(request_body))
        payload = json.dumps(data)
        endpoint_name = "bank-statements-ep-2022-06-23"
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType=content_type,
            Body=payload
        )
        result = json.loads(response['Body'].read().decode())['Output']
        return result

    @staticmethod
    def predict_ctos(feature_list_ctos):
        content_type = "application/json"
        request_body = {"Input": [feature_list_ctos]}
        data = json.loads(json.dumps(request_body))
        payload = json.dumps(data)
        endpoint_name = "ctos-ep-2022-06-23"
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType=content_type,
            Body=payload
        )
        result = json.loads(response['Body'].read().decode())['Output']
        return result

    def check_values(self):
        print(self.borrowerID)
        print(self.borrowerUEN)
        print(self.applicationData)

    def get_application_date(self):

        application_date = productionDBLoan.query_postgres('''
            SELECT
                LEFT(created_at::TEXT, 10) AS application_date
            FROM loans
            WHERE id=%(loanID)s;            
        ''', params={'loanID': self.loanID})['application_date'].values[0]

        return application_date

    def get_application_yrmth(self):

        application_yrmth = productionDBLoan.query_postgres('''
            SELECT
                -- LEFT(created_at::TEXT, 7) AS application_yrmth
                LEFT(updated_at::TEXT, 7) AS application_yrmth 
            FROM loans
            WHERE id=%(loanID)s;            
        ''', params={'loanID': self.loanID})['application_yrmth'].values[0]

        return application_yrmth

    # Check with DE if we have this in prod bd, otherwise take from GS
    def get_loan_application_quantum_tenor(self):

        df = productionDBLoan.query_postgres('''
            SELECT
                applied_amount,
                applied_tenor,
                tenor_type
            FROM loans
            WHERE id=%(loanID)s
        ''', params={'loanID': self.loanID})

        df['applied_tenor'] = np.where(df['tenor_type'] == 1, df['applied_tenor'] / 30, df['applied_tenor'])

        qt = {
            'appliedQuantum': df['applied_amount'].values[0],
            'appliedTenor': df['applied_tenor'].values[0]
        }

        return qt

    def get_historical_dpd(self):

        with snowflake.connector.connect(
                user=snowflake_cfg['username'],
                password=snowflake_cfg['password'],
                account=snowflake_cfg['host']
        ) as conn:

            conn.cursor().execute('USE WAREHOUSE DATA_SCIENCE');

            historical_dpd = pd.read_sql('''
                select LOAN_MAX_DPD_LCL_DAYS from adm.TRANSACTION.LOAN_DENORM_T where CMD_CTR_BORROWER_ID = %(borrower_id)s
                ''', params={'borrower_id': self.borrowerID},
                                         con=conn)
        historical_dpd.columns = [str.lower(col) for col in
                                  historical_dpd.columns]

        if historical_dpd.shape[0] == 0:
            return None
        else:
            return historical_dpd['loan_max_dpd_lcl_days'].max()

    def get_oth_params(self, item, format_numeric=True):
        """Helper function for getting params from google sheet
        Use format_numeric if numbers are required"""
        # print(self.othParams.to_string())
        sheet_item = self.othParams['ITEM'].str.strip().str.lower()
        item = item.strip().lower()
        assert self.othParams.loc[sheet_item == item].shape[0] == 1
        result = self.othParams.loc[sheet_item == item].values[0][1]
        print(result)
        # Changes to handle new EL_limit param
        if str(result).lower() == 'na':
            return None
        else:
            if format_numeric:
                result = float(result.replace(',', ''))
            return result

    def get_borrower_id(self):

        borrower_id = productionDBLoan.query_postgres('''
            SELECT borrower_id
            FROM loans
            WHERE id=%(loanID)s;
        ''', params={'loanID': self.loanID})['borrower_id'].values[0]

        return borrower_id

    # Check with DE
    # def get_entity_uen(self):
    #
    #     entity_uen = productionDBMember.query_postgres('''
    #           SELECT UPPER(TRIM(company_registration_number)) AS uen
    #           FROM members
    #           WHERE id=%(borrowerID)s
    #       ''', params={'borrowerID': str(self.borrowerID)})['uen'].values[0]
    #
    #     return entity_uen

    def get_historical_disbursed_loan_count(self):

        df = productionDBLoan.query_postgres('''
            SELECT borrower_id, id AS loan_id, created_at,
                ROW_NUMBER() OVER (
                    PARTITION BY borrower_id 
                        ORDER BY borrower_id, created_at ASC) - 1 
                            AS historical_disbursed_loan_count
            FROM loans
            WHERE status LIKE '%%SET%%' 
            AND borrower_id=%(borrowerID)s 
            AND created_at<%(applicationDate)s
        ''', params={'applicationDate': str(self.applicationDate), 'borrowerID': str(self.borrowerID)})

        if df.shape[0] <= 0:
            count = 0
        else:
            count = df['historical_disbursed_loan_count'].shape[0]

        return count

    def invoke(self):
        data = pd.DataFrame({
            'loanID': [int(self.loanID)],
            'borrowerID': [int(self.borrowerID)],
            'disbursedLoanCount': [self.disbursedLoanCount],
            'historicalDPD': [self.historicalDPD]
        })
        # TODO: Addend change as bs and ctos are DFs
        for info_dict in [self.loanQT, self.bankStatements, self.ctos_input]:
            for key, value in info_dict.items():
                data[key] = value
        data['bankCredit'] = float(self.bankStatements['monthly_credit'].values[0])
        data['bankLiabilities'] = float(self.bankStatements["monthly_liabilities"].values[0])
        return data

    def income_factor_scaling(self):
        # From credit underwriting's assessment of risk
        # print(self.industryRiskMap.info())
        # print("From DF", self.industryRiskMap['INDUSTRY'][16])
        # print(self.industry_sector)
        industry = str(self.industry_sector).strip().lower()
        income_scaling_factor = \
            int(self.industryRiskMap[self.industryRiskMap['INDUSTRY'] == industry]['INCOME_FACTOR'].values[0])
        try:
            risk_category = \
                self.industryRiskMap[self.industryRiskMap['INDUSTRY'] == industry]['RISK'].values[0]
        except:
            risk_category = None
        # print(income_scaling_factor, " ", risk_category)
        return income_scaling_factor, risk_category

    def prophesize(self):

        data = self.applicationData
        print("data\n", data)
        results = {}
        # self.industryCheck = data[
        #  ['industryRisk', 'industryBlacklistFlag', 'companyUEN', 'industrySector', 'industryDivision']]

        # Cap at internal quantum cap if applied quantum is higher

        results['applied_quantum'] = data['appliedQuantum'].values[0]
        # from LoanQT -> get_loan_application_quantum_tenor -> ProdDB Loans

        income_scaling_factor, risk_category = self.income_factor_scaling()
        print("Income Scale and Risk Category", income_scaling_factor, " ", risk_category)
        if risk_category is not None:
            self.riskQuantum = int(
                self.riskQuantumTenorMap[self.riskQuantumTenorMap["RISK"] == risk_category]["QUANTUM_CAP"].values[
                    0].replace(',', ''))
            self.riskTenor = int(
                self.riskQuantumTenorMap[self.riskQuantumTenorMap["RISK"] == risk_category]["TENOR_CAP"].values[0])
            print("Risk Quantum and Risk Tenor", self.riskQuantum, " ", self.riskTenor)
        results['risk_quantum'] = self.riskQuantum

        data['quantumCap'] = min(data['appliedQuantum'].values[0], self.riskQuantum)
        data['tenorCap'] = min(data['appliedTenor'].values[0], self.riskTenor)

        logger.info(f"applied quantum: {data['appliedQuantum'].values[0]}")
        logger.info(f"risk quantum: {self.riskQuantum}")
        logger.info(f"applied tenor: {data['appliedTenor'].values[0]}")
        logger.info(f"risk tenor: {self.riskTenor}")
        # data['quantumCap'] = np.where(
        #     (data['industryBlacklistFlag'] is True), 0, data['quantumCap'])
        # Changed as part of Nov 20 changes

        # data['TenorCap'] = np.where(
        #     (data['industryBlacklistFlag'] is True), 0, data['tenorCap'])
        results['quantum_cap'] = data['quantumCap'].values[0]
        results['tenor_cap'] = data['tenorCap'].values[0]
        logger.info(f"tenor cap: {data['tenorCap'].values[0]}")

        self.tenorCap = data['tenorCap'].values[0]
        # todo: check with the MY Credit team.
        data['Tenor'] = 6.0

        # bankQuantumCap = ((data['bankCredit'] - data['bankLiabilities']) * income_scaling_factor) * tenor_default
        print('bankCredit, bankLiabilities and Tenor', data['bankCredit'].values[0], data['bankLiabilities'].values[0],
              data['Tenor'].values[0])
        bankQuantumCap = (data['bankCredit'].values[0] * income_scaling_factor - data['bankLiabilities'].values[0]) * \
                         data['Tenor'].values[0]
        print("bankQuantumCap", bankQuantumCap)
        try:
            assert bankQuantumCap >= self.min_quantum
        except AssertionError as e:
            mod_err.raise_exception(str(e), ErrorLevel.WARNING,
                                    data=f"Bank quantum cap ({self.min_quantum}) hit: (credits * income_scaling - "
                                         f"liabilities) * tenor = ({data['bankCredit'].values[0]} * {bankQuantumCap} - "
                                         f"{data['bankLiabilities'].values[0]}) * 6 = {round(bankQuantumCap.values[0], 2)}")

        data['quantumCap'] = np.where(data['quantumCap'] > bankQuantumCap, bankQuantumCap, data['quantumCap'])
        data['quantumCap'] = np.where(data['quantumCap'] < 0, 0, data['quantumCap'])

        self.quantumCap = data['quantumCap'].values[0]
        results['data_quantum_cap'] = self.quantumCap
        # results['credit_7.5pct'] = data['bankCredit'].values[0] * income_scaling
        results['credit_income_scaling'] = data['bankCredit'].values[0] * income_scaling_factor

        results['liabilities'] = data['bankLiabilities'].values[0]
        results['diff'] = data['bankCredit'].values[0] * income_scaling_factor - data['bankLiabilities'].values[0]

        results['6*diff'] = 6 * (
                data['bankCredit'].values[0] * income_scaling_factor - data['bankLiabilities'].values[0])

        results['applied_tenor'] = data['appliedTenor'].values[0]
        results['risk_tenor'] = self.riskTenor
        # Mark: 28/1/21 Handling nans returned from bank statement parsing

        # Determine Underwriting Tenor Cap
        # data['tenorCap'] = np.where(
        #     (data['industryBlacklistFlag'] == True) | (data['industryRisk'] == 'HIGH'),
        #     0, 6)

        # Model input variables (replace with company type, etc)

        woeMaps = get_gsheet_values('1ohf_211FR14imGScxEowzsLNJ0ETlMz5PIt2Sta7MPA', 'WOE Table', "A1:E100")
        # print(woeMaps.head().to_string())
        woeMaps = woeMaps.reset_index(drop=True)
        woeMaps = woeMaps.replace('-444444444444', NINF)
        woeMaps = woeMaps.replace('333333333333', inf)

        woe_cols_bs = ["monthly_debit_pct", "dc_ratio_pct", "monthly_liabilities",
                       "bal_credit_ratio_pct", "bal_liab_ratio", "monthly_credit_pct",
                       "monthly_end_balance_pct", "bal_debit_ratio_pct"]

        woe_cols_ctos = ["onehotencoder__x1_ACTIVE", "onehotencoder__x0_nan",
                         "ccris_facility_arrears",
                         "ccris_application_total", "ccris_facility_total", "recourse_count",
                         "facility_value_normalised",
                         "total_recourse_ccris_application_count_approved",
                         "total_recourse_ccris_facility_count_arrears",
                         "ccris_application_approved", "onehotencoder__x0_PRIVATE LIMITED",
                         "onehotencoder__x1_EXISTING", "ccris_facility_value",
                         "recourse_ccris_application_count_total", "onehotencoder__x1_EXPIRED",
                         "onehotencoder__x1_nan", "cpy_years_of_establishment",
                         "onehotencoder__x0_SP", "onehotencoder__x0_PN"]

        def woeMapFN(data, woe_cols):
            """
            Map WOE values to feature bins.
            @param data: pandas.DataFrame: Feature set and woe_cols
            @return: pandas.DataFrame: Feature set replaced with WOE values
            """
            for col in woe_cols:
                data[col] = float(data[col])
                # print("col:", col)
                colValues = woeMaps[woeMaps['VARIABLE'].eq(col)]
                # print(colValues)
                condition = '[' + \
                            ','.join(colValues.apply(
                                lambda x: f"(data[col]>={(x['BIN_START'])}) & (data[col]<={(x['BIN_END'])})",
                                axis=1).values) + \
                            ']'
                # print("condition:", condition)
                values = colValues['WOE_VALUES']
                # print("values:", values)
                data[f'woe_{col}'] = np.select(eval(condition), values, default=np.nan)
            return data

        features_bs = data[woe_cols_bs]
        features_ctos = data[woe_cols_ctos]

        feature_list_bs = woeMapFN(features_bs, woe_cols_bs)
        feature_list_ctos = woeMapFN(features_ctos, woe_cols_ctos)
        # data = woeMapFN(data)

        feat_bs = ["woe_monthly_debit_pct",
                   "woe_dc_ratio_pct",
                   "woe_monthly_liabilities",
                   "woe_bal_credit_ratio_pct",
                   "woe_bal_liab_ratio",
                   "woe_monthly_credit_pct",
                   "woe_monthly_end_balance_pct",
                   "woe_bal_debit_ratio_pct"]

        feat_ctos = ["woe_onehotencoder__x1_ACTIVE",
                     "woe_onehotencoder__x0_nan",
                     "woe_ccris_facility_arrears",
                     "woe_ccris_application_total",
                     "woe_ccris_facility_total",
                     "woe_recourse_count",
                     "woe_facility_value_normalised",
                     "woe_total_recourse_ccris_application_count_approved",
                     "woe_total_recourse_ccris_facility_count_arrears",
                     "woe_ccris_application_approved",
                     "woe_onehotencoder__x0_PRIVATE LIMITED",
                     "woe_onehotencoder__x1_EXISTING",
                     "woe_ccris_facility_value",
                     "woe_recourse_ccris_application_count_total",
                     "woe_onehotencoder__x1_EXPIRED",
                     "woe_onehotencoder__x1_nan",
                     "woe_cpy_years_of_establishment",
                     "woe_onehotencoder__x0_SP",
                     "woe_onehotencoder__x0_PN",
                     ]
        feature_list_bs = feature_list_bs[feat_bs]
        feature_list_ctos = feature_list_ctos[feat_ctos]
        # print(feature_list_bs.to_string())
        # print(feature_list_ctos.to_string())
        print(feature_list_bs.values[0])
        print(feature_list_ctos.values[0])
        # mapped_features = woeMapping(woeMaps, data[woe_cols])
        # featureList = pandas.DataFrame(mapped_features)  # model features

        # Get QIT Matrix
        qit_matrix = []
        feature_list_bs = feature_list_bs.values[0]
        feature_list_ctos = feature_list_ctos.values[0]
        default_prob1 = self.predict_ctos(feature_list_ctos)
        default_prob2 = self.predict_bs(feature_list_bs)
        default_prob = (default_prob1 + default_prob2) / 2

        # default_prob = (default_prob1 + 2 * default_prob2) / 3
        for tenor in np.arange(1, 13):
            for quantum in [q for q in range(self.min_quantum, 10000, 1000)] + \
                           [q for q in range(10000, 25001, 5000)] + \
                           [q for q in range(30000, 100001, 10000)]:

                if (tenor <= self.tenorCap) & (quantum <= self.quantumCap):
                    calibrated_pd = default_prob
                    EAD, LGD = self.ead, self.lgd
                    EL = calibrated_pd * EAD * LGD

                    # Total Interest (fair odds)
                    total_interest = EL / (1 - EL)

                    # Monthly Interest
                    monthly_interest = round(total_interest / tenor, 4)

                    qit_matrix.append({
                        'quantum': int(quantum),
                        'tenor': int(tenor),
                        'total_interest': round(total_interest, 3),
                        'monthly_interest': monthly_interest,
                        'PD': calibrated_pd,
                    })
                else:
                    qit_matrix.append({
                        'quantum': int(quantum),
                        'tenor': int(tenor),
                        'total_interest': None,
                        'monthly_interest': None,
                        'PD': calibrated_pd,
                    })
        df_qit_matrix = pd.DataFrame(qit_matrix)
        df_qit_matrix_pivot = df_qit_matrix[['quantum', 'tenor', 'monthly_interest']].pivot(index='quantum',
                                                                                            columns='tenor')
        results['qit_matrix'] = df_qit_matrix_pivot.to_json()
        return results


def get_gsheet_values(sheet_id, sheet_name, sheet_range=''):
    scope = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/spreadsheets']
    credentials = service_account.Credentials.from_service_account_file(
        'client_secret.json', scopes=scope)
    SERVICE_NAME = 'sheets'
    API_VERSION = 'v4'
    SERVICE = discovery.build(SERVICE_NAME, API_VERSION, credentials=credentials)
    read_range = ("!".join([sheet_name, sheet_range]))
    result = SERVICE.spreadsheets().values().get(spreadsheetId=sheet_id, range=read_range).execute()
    rows = result['values']
    column_names = rows[0]
    m = re.compile('[^\w\d]+')
    column_names = [re.sub(m, '_', i).strip().upper() for i in column_names]
    df = pd.DataFrame(rows[1:], columns=column_names)
    return df


def calculate(loan_id):
    logger.info(f"loan_id:{loan_id}")
    # logger.info(f"is_bolt:{is_bolt}")

    # thorModel = pickle.load(open('model/thor_v6_gbdt_model.sav', 'rb'))
    # Updated Logistic Regression Model
    # thorModel = pickle.load(open('model/thor_v8.1_lr_model.sav', 'rb'))
    # calibrator = pickle.load(open('model/calibrator_v6_gbdt.sav', 'rb'))

    sheet_id = '1ohf_211FR14imGScxEowzsLNJ0ETlMz5PIt2Sta7MPA'
    industryRiskMap = get_gsheet_values(sheet_id, 'Credit Industry Risk Mapping', 'A1:C200')

    riskQuantumTenorMap = get_gsheet_values(sheet_id, 'Risk - Quantum & Tenor Mapping', 'A1:C100')

    othParams = get_gsheet_values(sheet_id, 'Other params', 'A1:C100')

    industryRiskMap["INDUSTRY"] = industryRiskMap["INDUSTRY"].map(lambda x: x if type(x) != str else x.lower())

    z = thor(industryRiskMap, riskQuantumTenorMap, othParams, loan_id)

    # loan_docs = get_loan_numbers_from_gsheets(loan_id)

    z.applicationData = z.invoke()

    # print(z.companyDetails)
    # z.Invoke()
    output = z.prophesize()

    # test = pd.read_json(output['qit_matrix'])
    # print(test)

    output['applied_quantum'] = float(output['applied_quantum'])
    output['risk_quantum'] = float(output['risk_quantum'])
    output['quantum_cap'] = float(output['quantum_cap'])
    output['credit_income_scaling'] = float(output['credit_income_scaling'])
    output['liabilities'] = float(output['liabilities'])
    output['diff'] = float(output['diff'])
    output['6*diff'] = float(output['6*diff'])
    output['applied_tenor'] = float(output['applied_tenor'])
    output['risk_tenor'] = float(output['risk_tenor'])
    output['tenor_cap'] = float(output['tenor_cap'])
    output['industry_risk'] = z.industryCheck.to_json()
    output['industry_black_list_flag'] = z.hardRejectCheck.to_json()
    output['company_details'] = z.companyDetails
    output['application_data'] = z.applicationData.to_json()

    # output_json = json.dumps(output)
    print(json.loads(output['application_data']))

    result = {}
    result['quantum'] = output['data_quantum_cap']
    # result['loan_code'] = loan_docs['loan_code']
    result['PD'] = output['PD']

    from ast import literal_eval as make_tuple
    qit_matrix = json.loads(output['qit_matrix'])
    qit_matrix_not_none = {k: v for k, v in qit_matrix.items() if v[str(z.min_quantum)] is not None}
    if len(list(qit_matrix_not_none.keys())) < 1:
        result['tenor'] = None
        result['rate'] = None
        result['quantum'] = None
        result['status'] = 'REJECTED'
        result['reason'] = 'Can\'t allocate quantum'
        result['PD'] = None
        write_to_rds(loan_id, result)
        return result
    qit_matrix_key = (list(qit_matrix_not_none.keys())[-1])
    months = make_tuple(qit_matrix_key)
    tenor = months[1]

    result['tenor'] = tenor
    filled_months = {k: v for k, v in qit_matrix[f"('monthly_interest', {int(result['tenor'])})"].items() if
                     v is not None}
    result['rate'] = filled_months[list(filled_months.keys())[-1]]
    result['status'] = 'ACCEPTED'
    result['reason'] = 'NONE'
    logger.info(result)
    write_to_rds(loan_id, result)
    return result


def json_load_test(json_file):
    from ast import literal_eval as make_tuple
    with open(json_file, 'r') as f:
        thor_json = json.loads(f.read())
        # dict_1 = {k:v for k,v in json.loads(thor_json['qit_matrix']).items() if v['3000'] != None}
        # key = (list(dict_1.keys())[-1])
        # tpl = make_tuple(key)
        # print(tpl[1])
        print(json.loads(thor_json['qit_matrix'])["('monthly_interest', 9)"])


def write_to_rds(loan_id, result):
    result['loan_id'] = loan_id
    result['error_message'] = str(mod_err.error_list)
    df = pd.DataFrame([{'loan_id': result['loan_id'], 'status': result['status'],
                        'PD': result['PD'], 'reason': result['reason'], 'quantum': result['quantum'],
                        'tenor': result['tenor'], 'rate': result['rate'], 'error_message': result['error_message']}])
    # 'loan_code': result['loan_code'],
    productionRDSResults.write_to_postgres(df, 'append', False, 'public', 'bolt_MY_results')


if __name__ == "__main__":
    calculate(2611539)
    print(mod_err.error_list)

# print(mod_err.error_list)
# json_load_test('thor.json')
# sys.exit()
