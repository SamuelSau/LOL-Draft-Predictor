import os
import unittest
from dotenv import load_dotenv
from src import retrieve_games_data

load_dotenv()
api_key = os.getenv("API_KEY") or ""

# Limit test scope
TEST_REGION = "na1"
TEST_TIER = "MASTER"
TEST_DIVISION = "I"
TEST_PAGE_INDEX = 1  # Only 1 page

class TestRetrieveGamesDataIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.puuids_file = "test_puuids.txt"
        cls.match_ids_file = "test_match_ids.txt"
        cls.all_puuids_region = set()
        cls.all_match_ids_region = set() 

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.puuids_file):
            os.remove(cls.puuids_file)
        if os.path.exists(cls.match_ids_file):
            os.remove(cls.match_ids_file)

    def test_collect_puuids_sample(self):
        print(f"Sampling players from {TEST_REGION}/{TEST_TIER}/{TEST_DIVISION}...")

        url = f"https://{TEST_REGION}.api.riotgames.com/lol/league-exp/v4/entries/RANKED_SOLO_5x5/{TEST_TIER}/{TEST_DIVISION}?page={TEST_PAGE_INDEX}&api_key={api_key}"
        headers = {"X-Riot-Token": api_key}
        response = retrieve_games_data.requests.get(url, headers=headers)

        self.assertEqual(response.status_code, 200)
        players = response.json()

        sampled_players = players[:3]  # Limit to 3 for testing
        for player in sampled_players:
            puuid = player.get("puuid")
            if puuid:
                self.__class__.all_puuids_region.add((puuid, TEST_REGION))

        # Write to test file
        with open(self.puuids_file, "w") as f:
            for puuid, region in self.all_puuids_region:
                f.write(f"{puuid},{region}\n")

        self.assertGreater(len(self.all_puuids_region), 0)

    def test_collect_match_ids_sample(self):
        result = retrieve_games_data.collect_match_ids_and_write_to_file(
            api_key,
            self.match_ids_file,
            self.__class__.all_puuids_region,
            self.__class__.all_match_ids_region
        )
        self.__class__.all_match_ids_region = result
        self.assertGreater(len(result), 0)

    def test_collect_match_details_sample(self):
        sample_ids = list(self.__class__.all_match_ids_region)[:2]  # Sample only 2 matches
        region = sample_ids[0][1]
        result = retrieve_games_data.collect_match_details(api_key, region, set(sample_ids))

        self.assertGreater(len(result), 0)
        self.assertIn("participants", result[0])
        self.assertIn("teams", result[0])
        self.assertIn("match_id", result[0])

if __name__ == "__main__":
    unittest.main()
