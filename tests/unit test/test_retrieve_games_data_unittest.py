import os
import sys
import unittest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import retrieve_games_data

class TestRetrieveGamesData(unittest.TestCase):

    def setUp(self):
        self.sample_match_ids = [("NA_match1", "na1")]
        self.sample_puuids = {("puuid1", "na1")}

        self.sample_match_details = {
            "info": {
                "gameVersion": "13.1.1",
                "teams": [{"teamId": 100, "win": True}, {"teamId": 200, "win": False}],
                "participants": [
                    {"championId": 1, "championName": "Annie", "teamId": 100, "teamPosition": "MID", "win": True},
                    {"championId": 2, "championName": "Olaf", "teamId": 200, "teamPosition": "TOP", "win": False}
                ]
            }
        }

        self.expected_processed_match = {
            'match_id': 'NA_match1',
            'game_version': '13.1.1',
            'teams': [{'team_id': 100, 'win': True}, {'team_id': 200, 'win': False}],
            'participants': [
                {'champion_id': 1, 'champion_name': 'Annie', 'team_id': 100, 'team_position': 'MID', 'win': True},
                {'champion_id': 2, 'champion_name': 'Olaf', 'team_id': 200, 'team_position': 'TOP', 'win': False}
            ]
        }

    @patch('src.retrieve_games_data.requests.get')
    @patch('src.retrieve_games_data.time.sleep')
    @patch('builtins.open', new_callable=mock_open)
    def test_collect_puuids_and_write_to_file(self, mock_file, mock_sleep, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"puuid": "puuid1"}]
        mock_get.return_value = mock_response

        result = retrieve_games_data.collect_puuids_and_write_to_file("fake_api_key", "dummy_path.txt", set())
        self.assertIn(("puuid1", "na1"), result)

    @patch('src.retrieve_games_data.requests.get')
    @patch('src.retrieve_games_data.time.sleep')
    @patch('builtins.open', new_callable=mock_open)
    def test_collect_match_ids_and_write_to_file(self, mock_file, mock_sleep, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = ["NA_match1"]
        mock_get.return_value = mock_response

        result = retrieve_games_data.collect_match_ids_and_write_to_file(
            "fake_api_key", "dummy_path.txt", {("puuid1", "na1")}, set()
        )
        self.assertIn(("NA_match1", "na1"), result)

    @patch('src.retrieve_games_data.requests.get')
    def test_collect_match_details(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_match_details
        mock_get.return_value = mock_response

        results = retrieve_games_data.collect_match_details("fake_api_key", "na1", set(self.sample_match_ids))
        self.assertEqual(results[0]['match_id'], self.expected_processed_match['match_id'])

    @patch('src.retrieve_games_data.parse_game')
    @patch('src.retrieve_games_data.np.savez_compressed')
    @patch('os.path.exists')
    def test_save_match_details_to_npz_new_file(self, mock_exists, mock_savez, mock_parse_game):
        mock_exists.return_value = False
        mock_parse_game.return_value = (np.array([1, 2, 3], dtype=np.float32), np.array([1], dtype=np.float32))
        
        retrieve_games_data.save_match_details_to_npz([self.expected_processed_match])

        mock_parse_game.assert_called_once()
        mock_savez.assert_called_once()
        args, kwargs = mock_savez.call_args
        self.assertIn('X', kwargs)
        self.assertIn('y', kwargs)

    def test_get_routing_value(self):
        self.assertEqual(retrieve_games_data.get_routing_value("na1"), "americas")
        self.assertEqual(retrieve_games_data.get_routing_value("euw1"), "europe")
        self.assertEqual(retrieve_games_data.get_routing_value("kr"), "asia")
        self.assertEqual(retrieve_games_data.get_routing_value("unknown"), "americas")  # default case

if __name__ == "__main__":
    unittest.main()
