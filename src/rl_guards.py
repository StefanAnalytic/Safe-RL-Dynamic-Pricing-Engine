import numpy as np

class SafePricingGuard:
    def __init__(self, cost):
        self.cost = cost
        
    def safe_action_wrapper(self, proposed_action_price, comp_price, inventory):
        """
        Der vom Boss geforderte Kill-Switch für das Actionability Gate.
        """
        # Fault 2: Negative Inventory Sync Issue
        if inventory <= 0:
            # Kein Bestand -> Maximaler Preis, um Nachfrage zu killen
            return comp_price * 1.5 
            
        # Fault 1: NaN Competitor Price Handling (API Outage)
        if np.isnan(comp_price):
            safe_comp_price = self.cost * 2.0
        else:
            safe_comp_price = comp_price
        
        # KILL-SWITCH / ACTION CLIPPING (Safe RL)
        min_price = self.cost * 1.10  # Harter Margin Floor (niemals unter 10% Marge)
        max_price = safe_comp_price * 1.05  # Ceiling (nicht mehr als 5% über Konkurrenz)
        
        # Tensor-Ebene: Wir zwingen den Preis in den erlaubten Korridor
        executed_price = np.clip(proposed_action_price, min_price, max_price)
        
        return executed_price