import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from templates import MENU_ORDER, SUPPORTED_MARKETS, default_market_message, market_button_text


class TemplateTests(unittest.TestCase):
    def test_menu_has_required_markets_in_order(self):
        self.assertEqual(MENU_ORDER, ("GOLD", "SILVER", "BTC", "ETH"))
        self.assertEqual(set(MENU_ORDER), set(SUPPORTED_MARKETS))

    def test_button_text_includes_human_label(self):
        self.assertEqual(market_button_text("GOLD"), "🥇 Gold")
        self.assertEqual(market_button_text("ETH"), "Ξ Ethereum")

    def test_default_message_identifies_market(self):
        message = default_market_message("BTC")
        self.assertIn("Bitcoin", message)
        self.assertIn("BTC", message)
        self.assertIn("Signal template is ready", message)


if __name__ == "__main__":
    unittest.main()
