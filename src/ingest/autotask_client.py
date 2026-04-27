"""
Autotask REST API client — stub for future implementation.
API key and zone URL will be required once access is provisioned.

Autotask REST API docs: https://ww1.autotask.net/help/developerhelp/Content/APIs/REST/REST_API.htm
"""


class AutotaskClient:
    def __init__(self, api_key: str, api_secret: str, zone_url: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.zone_url = zone_url  # e.g. https://webservices4.autotask.net

    def fetch_completed_tickets(self, days_back: int = 30):
        raise NotImplementedError(
            "Autotask API access not yet provisioned. Use CSV import for now."
        )
