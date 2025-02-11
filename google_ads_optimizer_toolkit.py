from superagi.tools.base_tool import BaseToolkit
from google_ads_optimizer_tool import GoogleAdsOptimizer

class GoogleAdsOptimizerToolkit(BaseToolkit):
    name = "Google Ads Optimizer Toolkit"
    description = "Toolkit for optimizing Google Ads campaigns based on sales data."

    def get_tools(self):
        return [GoogleAdsOptimizer()]

    def get_env_keys(self):
        """
        Возвращает список ключей, которые будут отображены во frontend SuperAGI
        для настройки инструмента.
        """
        return [
            {
                "name": "GOOGLE_ADS_CLIENT_ID",
                "description": "Client ID для Google Ads API",
                "default": "",
                "type": "string",
                "required": True
            },
            {
                "name": "GOOGLE_ADS_CLIENT_SECRET",
                "description": "Client Secret для Google Ads API",
                "default": "",
                "type": "string",
                "required": True
            },
            {
                "name": "GOOGLE_ADS_REFRESH_TOKEN",
                "description": "Refresh Token для Google Ads API",
                "default": "",
                "type": "string",
                "required": True
            },
            {
                "name": "GOOGLE_ADS_DEVELOPER_TOKEN",
                "description": "Developer Token для Google Ads API",
                "default": "",
                "type": "string",
                "required": True
            },
            {
                "name": "GOOGLE_ADS_LOGIN_CUSTOMER_ID",
                "description": "Login Customer ID для управления аккаунтами Google Ads",
                "default": "",
                "type": "string",
                "required": False
            },
            {
                "name": "DATABASE_URL",
                "description": "URL для подключения к базе данных MySQL",
                "default": "mysql+pymysql://user:password@host:port/dbname",
                "type": "string",
                "required": True
            }
        ]
