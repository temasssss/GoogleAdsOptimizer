from superagi.tools.base_tool import BaseToolkit
from google_ads_optimizer_tool import GoogleAdsOptimizer

class GoogleAdsOptimizerToolkit(BaseToolkit):
    name = "Google Ads Optimizer Toolkit"
    description = "Toolkit for optimizing Google Ads campaigns based on sales data."

    def get_tools(self):
        return [GoogleAdsOptimizer()]

    def get_env_keys(self):
        return [
            "GOOGLE_ADS_CLIENT_ID",
            "GOOGLE_ADS_CLIENT_SECRET",
            "GOOGLE_ADS_REFRESH_TOKEN",
            "GOOGLE_ADS_DEVELOPER_TOKEN",
            "GOOGLE_ADS_LOGIN_CUSTOMER_ID",
            "DATABASE_URL"
        ]
