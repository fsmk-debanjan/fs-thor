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
# import psycopg

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
        except Exception as e:
            mod_err.raise_exception(str(e), ErrorLevel.WARNING,
                                    data=f'Failed to get the Bank statements and the Ctos data for loan ID: {self.loanID}')
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
        # Separating both the Inputs
        df_bank = df.loc[:, "AVGBANKMONTHENDBALANCES":]
        df_ctos = df.loc[:, :"TOTALRECOURSECCRISAPPLICATIONCOUNT"]
        industry_sector = df.loc[:, "INDUSTRYSECTOR"]
        # print(df_bank.to_string())
        # print("\n\n\n\n\n")
        # print(df_ctos.to_string())
        return self.transform_bs(df_bank), self.transform_ctos(df_ctos), industry_sector

    @staticmethod
    def transform_bs(df_bank):
        df_bank_transform = pd.DataFrame(columns=["monthly_debit_ptc", "dc_ratio_pct", "monthly_liabilities",
                                                  "bal_credit_ratio_pct", "bal_liab_ratio", "monthly_credit_pct",
                                                  "monthly_end_balance_pct", "bal_debit_ratio_pct"])
        df_bank_transform["monthly_debit_ptc"] = df_bank["BANKDEBITPCT"]
        df_bank_transform["dc_ratio_pct"] = df_bank["DCRATIOPCT"]
        df_bank_transform["monthly_liabilities"] = df_bank["BANKLIABILITIES"]
        df_bank_transform["bal_credit_ratio_pct"] = df_bank["BALANCECREDITRATIOPCT"]
        df_bank_transform["bal_liab_ratio"] = df_bank["BALANCELIABILITYRATIO"]
        df_bank_transform["monthly_credit_pct"] = df_bank["BANKCREDITPCT"]
        df_bank_transform["monthly_end_balance_pct"] = df_bank["BANKENDBALANCESPCT"]
        df_bank_transform["bal_debit_ratio_pct"] = df_bank["BALANCEDEBITRATIOPCT"]
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
            company_status.get(df_ctos["X"].values[0].upper(), [0, 0, 0, 1])[0]
        df_ctos_transform["onehotencoder__x0_nan"] = company_type.get(df_ctos["COMPANYTYPE"].values[0], [1, 0, 0, 0])[0]

        df_ctos_transform["ccris_facility_arrears"] = df_ctos["CCRISFACILITYARREARS_1_0_"]  #
        df_ctos_transform["ccris_application_total"] = df_ctos["CCRISAPPLICATIONTOTALCOUNT"]  #
        df_ctos_transform["ccris_facility_total"] = df_ctos["CCRISOSTDFACILITYCOUNT"]  #
        df_ctos_transform["recourse_count"] = df_ctos["CCRISAPPLICATIONAPPROVEDCOUNT"]  #
        df_ctos_transform["facility_value_normalised"] = np.double(df_ctos['CCRISOSTDFACILITYVALUE']) / np.double(
            df_ctos['CCRISOSTDFACILITYCOUNT'])

        df_ctos_transform["total_recourse_ccris_application_count_approved"] = df_ctos[
            "TOTALRECOURSECCRISAPPLICATIONAPPROVEDCOUNT"]  #
        df_ctos_transform["total_recourse_ccris_facility_count_arrears"] = df_ctos[
            "TOTALRECOURSECOUNTCCRISFACILITYARREARS"]  #
        df_ctos_transform["ccris_application_approved"] = df_ctos["RECOURSECOUNT"]  #

        # fill-2
        df_ctos_transform["onehotencoder__x0_PRIVATE LIMITED"] = \
            company_type.get(df_ctos["COMPANYTYPE"].values[0], [1, 0, 0, 0])[1]
        df_ctos_transform["onehotencoder__x1_EXISTING"] = \
            company_status.get(df_ctos["X"].values[0].upper(), [0, 0, 0, 1])[1]

        df_ctos_transform["ccris_facility_value"] = df_ctos["CCRISOSTDFACILITYVALUE"]  #
        df_ctos_transform["recourse_ccris_application_count_total"] = df_ctos["TOTALRECOURSECCRISAPPLICATIONCOUNT"]  #

        # fill-3
        df_ctos_transform["onehotencoder__x1_EXPIRED"] = \
            company_status.get(df_ctos["X"].values[0].upper(), [0, 0, 0, 1])[2]
        df_ctos_transform["onehotencoder__x1_nan"] = \
            company_status.get(df_ctos["X"].values[0].upper(), [0, 0, 0, 1])[3]

        df_ctos_transform["cpy_years_of_establishment"] = df_ctos["YEARSOFESTABLISHMENT"]  #

        # fill-4
        df_ctos_transform["onehotencoder__x0_SP"] = \
            company_type.get(df_ctos["COMPANYTYPE"].values[0], [1, 0, 0, 0])[2]
        df_ctos_transform["onehotencoder__x0_PN"] = \
            company_type.get(df_ctos["COMPANYTYPE"].values[0], [1, 0, 0, 0])[3]

        df_ctos_transform = df_ctos_transform.replace(np.inf, 333333333333)
        df_ctos_transform = df_ctos_transform.replace(-np.inf, -444444444444)
        df_ctos_transform = df_ctos_transform.fillna(999999999999)
        return df_ctos_transform

    def predict_bs(self):
        runtime_client = boto3.client('sagemaker-runtime')
        content_type = "application/json"
        input1 = self.bankStatements
        request_body = {"Input": [[0.479327, 0.435842, 0.041859, 0.295023, 0.042205, -0.195128, 0.138722, 0.789482]]}
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

    def predict_ctos(self):
        runtime_client = boto3.client('sagemaker-runtime')
        content_type = "application/json"
        request_body = {"Input": [self.ctos_input]}
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
        model_field = 'bolt'

        sheet_item = self.othParams.item.str.strip().str.lower()
        item = item.strip().lower()
        assert self.othParams.loc[sheet_item == item, model_field].shape[0] == 1
        result = self.othParams.loc[sheet_item == item, model_field].iloc[0]
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

        return data

    def income_factor_scaling(self):
        # From credit underwriting's assessment of risk
        income_scaling_factor = \
            self.riskQuantumTenorMap[self.riskQuantumTenorMap['INDUSTRY'] == self.industry_sector][
                'INCOME_FACTOR'].values[0]
        risk_category = \
            self.riskQuantumTenorMap[self.riskQuantumTenorMap['INDUSTRY'] == self.industry_sector]['RISK'].values[0]
        return income_scaling_factor, risk_category

    def prophesize(self):

        data = self.applicationData
        results = {}
        # self.industryCheck = data[
        #  ['industryRisk', 'industryBlacklistFlag', 'companyUEN', 'industrySector', 'industryDivision']]

        # Cap at internal quantum cap if applied quantum is higher

        results['applied_quantum'] = data['appliedQuantum'].values[0]  # from LoanQT -> get_loan_application_quantum_tenor -> ProdDB Loans

        income_scaling_factor, risk_category = self.income_factor_scaling()
        if risk_category is not None:
            self.riskQuantum = int(self.riskQuantumTenorMap["RISK" == risk_category]["QUANTUM_CAP"].replace(',', ''))
            self.riskTenor = int(self.riskQuantumTenorMap["RISK" == risk_category]["TENOR_CAP"])
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
        data['Tenor'] = 6.0

        # bankQuantumCap = ((data['bankCredit'] - data['bankLiabilities']) * income_scaling_factor) * tenor_default

        bankQuantumCap = (data['bankCredit'] * income_scaling_factor - data['bankLiabilities']) * data['Tenor']

        try:
            assert bankQuantumCap.values[0] >= self.min_quantum
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
        results['6*diff'] = 6 * (data['bankCredit'].values[0] * income_scaling_factor - data['bankLiabilities'].values[0])

        results['applied_tenor'] = data['appliedTenor'].values[0]
        results['risk_tenor'] = self.riskTenor
        # Mark: 28/1/21 Handling nans returned from bank statement parsing

        # Determine Underwriting Tenor Cap
        # data['tenorCap'] = np.where(
        #     (data['industryBlacklistFlag'] == True) | (data['industryRisk'] == 'HIGH'),
        #     0, 6)

        # Model input variables (replace with company type, etc)

        woeMaps = get_gsheet_values('1ohf_211FR14imGScxEowzsLNJ0ETlMz5PIt2Sta7MPA', 'WOE Table', "A1:E100")

        woeMaps = woeMaps.reset_index(drop=True)
        woeMaps = woeMaps.replace('-999', NINF)
        woeMaps = woeMaps.replace('999', inf)

        woe_cols = ["monthly_debit_ptc", "dc_ratio_pct", "monthly_liabilities",
                    "bal_credit_ratio_pct", "bal_liab_ratio", "monthly_credit_pct",
                    "monthly_end_balance_pct", "bal_debit_ratio_pct",
                    "onehotencoder__x1_ACTIVE", "onehotencoder__x0_nan",
                    "ccris_facility_arrears",
                    "ccris_application_total", "ccris_facility_total", "recourse_count",
                    "facility_value_normalised",
                    "total_recourse_ccris_application_count_approved",
                    "total_recourse_ccris_facility_count_arrears",
                    "ccris_application_approved", "onehotencoder__x0_PRIVATE LIMITED",
                    "onehotencoder__x1_EXISTING", "ccris_facility_value",
                    "recourse_ccris_application_count_total", "onehotencoder__x1_EXPIRED",
                    "onehotencoder__x1_nan", "cpy_years_of_establishment",
                    "onehotencoder__x0_SP", "onehotencoder__x0_PN"
                    ]

        def woeMapFN(data):
            """
            Map WOE values to feature bins.
            @param data: pandas.DataFrame: Feature set
            @return: pandas.DataFrame: Feature set replaced with WOE values
            """
            for col in woe_cols:
                colValues = woeMaps[woeMaps['variable'].eq(col)]
                condition = '[' + \
                            ','.join(colValues.apply(
                                lambda x: f"(data[col]>={x['bin_start']}) & (data[col]<={x['bin_end']})",
                                axis=1).values) + \
                            ']'
                values = colValues['woe_values']
                data[f'woe_{col}'] = np.select(eval(condition), values, default=np.nan)
            return data

        data = woeMapFN(data)

        feat_bs = ["monthly_debit_ptc", "dc_ratio_pct", "monthly_liabilities",
                   "bal_credit_ratio_pct", "bal_liab_ratio", "monthly_credit_pct",
                   "monthly_end_balance_pct", "bal_debit_ratio_pct"]

        feat_ctos = ["onehotencoder__x1_ACTIVE", "onehotencoder__x0_nan",
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

        feature_list_bs = data[feat_bs]
        feature_list_ctos = data[feat_ctos]

        # mapped_features = woeMapping(woeMaps, data[woe_cols])
        # featureList = pandas.DataFrame(mapped_features)  # model features

        # Get QIT Matrix
        qit_matrix = []
        default_prob1 = self.predict_ctos(feature_list_ctos)
        default_prob2 = self.predict_bs(feature_list_bs)
        default_prob = (default_prob1 +  default_prob2) / 2

        for tenor in np.arange(1, 13):
            for quantum in [q for q in range(self.min_quantum, 10000, 1000)] + \
                           [q for q in range(10000, 25001, 5000)] + \
                           [q for q in range(30000, 100001, 10000)]:

                if (tenor <= self.tenorCap) & (quantum <= self.quantumCap):
                    # Predict Raw Model PD
                    # Extract PD from your function

                    # default_prob = self.thorModel.predict_proba(model_input)[:, 1][0]
                    # pd = self.thorModel.predict(model_input)[0]
                    # Calibrate Raw Model PD
                    # calibrated_pd = self.calibrator.predict(np.array([pd]))[0]

                    # Bypass the calibrator since Logistic Regression model is pre-calibrated
                    calibrated_pd = default_prob
                    # Compute avg. PD based on CTOS and Bank statements

                    # # EL where LGD assumed to be 1
                    # if self.grantFlag:
                    #     # EAD = 1
                    #     EAD = 0.65  # Grant is no longer a bullet repayment
                    # else:
                    #     EAD = 0.65
                    EAD, LGD = self.ead, self.lgd
                    EL = calibrated_pd * EAD * LGD

                    # Total Interest (fair odds)
                    total_interest = EL / (1 - EL)

                    # Term Conversion
                    # Don't have term conversion
                    # total_interest = total_interest * self.termConversion[tenor]

                    # # Profit Loading
                    # total_interest += self.termConversion[tenor] * 0.0075
                    #
                    # # Cost of Funds
                    # total_interest += self.termConversion[tenor] * 0.005

                    # Monthly Interest
                    monthly_interest = round(total_interest / tenor, 4)

                    # Mark: Removing logic as requested by credit underwriting
                    if self.int_limit is None or monthly_interest < self.int_limit:
                        # logger.info(f'monthly_interest: {monthly_interest})')
                        pass
                        # Applying Floor
                        # if self.int_floor is not None:
                        #     if monthly_interest < self.int_floor:
                        #         monthly_interest = self.int_floor
                    else:
                        # Mark: Removing logic 1/4/2020
                        # logger.warning(f'Monthly int limit exceeded :{quantum} for {tenor}mths @ {monthly_interest}')
                        # monthly_interest = None
                        pass

                    # if monthly_interest < 0.015:
                    #     monthly_interest = 0.015
                    # ESG grant financing ESG
                    # if monthly_interest < 0.01:
                    #     monthly_interest = 0.01

                    # Applying Cap
                    # if monthly_interest > 0.05:
                    #     monthly_interest = None

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


def get_gsheet_values(sheets_service, spreadsheetID, range):
    """Helper function for creating dataframes from google sheets"""
    sheet = sheets_service.spreadsheets().values().get(
        spreadsheetId=spreadsheetID,
        range=range).execute()
    df = pd.DataFrame(sheet['values'])
    columns = df.iloc[0, :].values
    columns = [str.lower(col) for col in columns]
    df = df.iloc[1:, :]
    df.columns = columns
    return df


def get_gsheet_values_from_model_port(sheets_service, spreadsheetID, range):
    """Helper function for creating dataframes from google sheets"""
    sheet = sheets_service.spreadsheets().values().get(
        spreadsheetId=spreadsheetID,
        range=range).execute()
    df = pd.DataFrame(sheet['values'])
    return df[2:]


# NOTE TO DEV: Get the GS values. Can replace with your own function.
def get_loan_numbers_from_gsheets(loan_id, spreadsheetID='1-q_obW7b_WGEmJIz_VknnVBlyQ-BUGVVRZ0qnpCewtM'):
    with open('config/google_sheet_creds.pickle', 'rb') as token:
        creds = pickle.load(token)
        sheets_service = build('sheets', 'v4', credentials=creds)

        # CHANGELOG: Date:20210820, Author: Debanjan, Desc: Updated g-sheet with ratios and pct values.

        loan_sheet = get_gsheet_values_from_model_port(sheets_service, '1-q_obW7b_WGEmJIz_VknnVBlyQ-BUGVVRZ0qnpCewtM',
                                                       'Model Port (NEW - thor 8.0)')

        loan_sheet = loan_sheet.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                         11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                         21, 22, 27, 28, 31, 34, 35, 36, 37, 38,
                                         39, 40, 41, 42, 43, 44, 45, 46, 47, 48]]

        loan_sheet.columns = ['sn', 'loan_id', 'loan_code', 'borrower',
                              'bucket', 'uen_number', 'entity_type',
                              'ssic_code', 'industry_sector', 'cbs_risk_grade',
                              'cbs_bankruptcy_proceedings', 'cbs_outstanding_defaults',
                              'cbs_outstanding_unsecured_balances',
                              'bri_risk_grade', 'bri_pending_litigation',
                              'bri_bankruptcy_l3y', 'bri_ongoing_money_lender_loan', 'brc_risk_grade', 'entity_age',
                              'brc_pending_litigation', 'brc_ongoing_money_lender_loans',
                              'noa_income', 'bank_end_balances', 'bank_liabilities', 'bank_credit', 'bank_debit',
                              'bank_statement_months',
                              'dc_ratio', 'bal_credit_ratio', 'bal_debit_ratio', 'bal_liability_ratio',
                              'bank_credit_pct', 'bank_debit_pct', 'bank_end_balances_pct',
                              'bank_liabilities_pct', 'dc_ratio_pct', 'bal_debit_ratio_pct', 'bal_credit_ratio_pct',
                              'bal_liability_ratio_pct', 'disbursed_loan_count', 'historical_dpd']

        loan_sheet = loan_sheet[loan_sheet['loan_id'] == str(loan_id)]
        loan_sheet = loan_sheet.tail(1)  # Take the last value for multiple loan IDs
        loan_sheet = loan_sheet.replace('#N/A', np.nan)

        # Don't have CBS, BRC, BRI
        #
        # cbs = {}
        # cbs['cbsRiskGrade'] = loan_sheet['cbs_risk_grade'].values[0]
        # cbs['cbsBankruptcyProceedings'] = int(loan_sheet['cbs_bankruptcy_proceedings'].values[0])
        # cbs['cbsOutstandingDefaults'] = int(loan_sheet['cbs_outstanding_defaults'].values[0])
        # cbs['cbsOutstandingUnsecuredBalances'] = float(loan_sheet['cbs_outstanding_unsecured_balances'].values[0])
        #
        # bri = {}
        # bri['briRiskGrade'] = loan_sheet['bri_risk_grade'].values[0]
        # bri['briPendingLitigation'] = int(loan_sheet['bri_pending_litigation'].values[0])
        # bri['briBankruptcyL3y'] = int(loan_sheet['bri_bankruptcy_l3y'].values[0])
        # bri['briOngoingMoneylenderLoans'] = int(loan_sheet['bri_ongoing_money_lender_loan'].values[0])
        #
        # brc = {}
        # brc['brcRiskGrade'] = loan_sheet['brc_risk_grade'].values[0]
        # brc['entityAge'] = int(loan_sheet['entity_age'].values[0])
        # brc['brcPendingLitigation'] = int(loan_sheet['brc_pending_litigation'].values[0])
        # brc['brcOngoingMoneylenderLoans'] = int(loan_sheet['brc_ongoing_money_lender_loans'].values[0])

        # CHANGELOG: Date:20210820, Author: Debanjan, Desc: Added ratios and percentage changes
        bank_statements = {}
        bank_statements['bankEndBalances'] = float(loan_sheet['bank_end_balances'].values[0])
        bank_statements['bankLiabilities'] = float(loan_sheet['bank_liabilities'].values[0])
        bank_statements['bankCredit'] = float(loan_sheet['bank_credit'].values[0])
        bank_statements['bankDebit'] = float(loan_sheet['bank_debit'].values[0])
        bank_statements['DCRatio'] = float(loan_sheet['dc_ratio'].values[0])
        bank_statements['balanceCreditRatio'] = float(loan_sheet['bal_credit_ratio'].values[0])
        bank_statements['balanceDebitRatio'] = float(loan_sheet['bal_debit_ratio'].values[0])
        bank_statements['balanceLiabilityRatio'] = float(loan_sheet['bal_liability_ratio'].values[0])
        bank_statements['bankStatementMonths'] = int(loan_sheet['bank_statement_months'].values[0])
        bank_statements['bankCreditPct'] = float(loan_sheet['bank_credit_pct'].values[0])
        bank_statements['bankDebitPct'] = float(loan_sheet['bank_debit_pct'].values[0])
        bank_statements['bankEndBalancesPct'] = float(loan_sheet['bank_end_balances_pct'].values[0])
        bank_statements['bankLiabilitiesPct'] = float(loan_sheet['bank_liabilities_pct'].values[0])
        bank_statements['balanceCreditRatioPct'] = float(loan_sheet['bal_credit_ratio_pct'].values[0])
        bank_statements['balanceDebitRatioPct'] = float(loan_sheet['bal_debit_ratio_pct'].values[0])
        bank_statements['balanceLiabilityRatioPct'] = float(loan_sheet['bal_liability_ratio_pct'].values[0])
        bank_statements['DCRatioPct'] = float(loan_sheet['dc_ratio_pct'].values[0])

        company_details = {}
        company_details['entity_type'] = loan_sheet['entity_type'].values[0]
        company_details['ssic_code'] = loan_sheet['ssic_code'].values[0]
        company_details['industry_group'] = ''
        company_details['industry_division'] = ''
        company_details['industry_sector'] = loan_sheet['industry_sector'].values[0]

        noa_income = {}
        noa_income['noaIncome'] = float(loan_sheet['noa_income'].values[0])

        loan = {}
        loan['loan_id'] = loan_sheet['loan_id'].values[0]
        loan['loan_code'] = loan_sheet['loan_code'].values[0]
        loan['bucket'] = loan_sheet['bucket'].values[0]
        loan['bank_statements'] = bank_statements
        loan['company_details'] = company_details
        loan['noa_income'] = noa_income

        return loan


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
    output['noaIncome'] = float(output['noaIncome'])
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
    result['loan_code'] = loan_docs['loan_code']
    result['PD'] = output['PD']

    from ast import literal_eval as make_tuple
    qit_matrix = json.loads(output['qit_matrix'])
    qit_matrix_not_none = {k: v for k, v in qit_matrix.items() if v[str(z.min_quantum)] is not None}
    if len(list(qit_matrix_not_none.keys())) < 1:
        result['tenor'] = None
        result['rate'] = None
        result['quantum'] = None
        result['status'] = 'REJECTED'
        result['reason'] = 'Cant allocate quantum'
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
    df = pd.DataFrame([{'loan_id': result['loan_id'], 'loan_code': result['loan_code'], 'status': result['status'],
                        'PD': result['PD'], 'reason': result['reason'], 'quantum': result['quantum'],
                        'tenor': result['tenor'],'rate': result['rate'], 'error_message': result['error_message']}])
    productionRDSResults.write_to_postgres(df, 'append', False, 'public', 'bolt_MY_results')


if __name__ == "__main__":
    calculate(3030682)
    print(mod_err.error_list)

# print(mod_err.error_list)
# json_load_test('thor.json')
# sys.exit()
