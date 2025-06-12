import pandas as pd
import requests
import threading
import time
import streamlit as st

import hashlib
import json
import openai
import os
from datetime import datetime

# Set OpenAI API key (store securely in environment variables)
# os.environ["OPENAI_API_KEY"] = "your-api-key"

class DataLoader:
    def __init__(self):
        self.data_idx = 0
        self.interval= 30 #second for data checks
        self.data = pd.DataFrame()
        self.new_data = pd.DataFrame()
    def _fetch_data(self):
        try:
            with 
            if new_data.empty:
                return
                
            with self.lock:
                if self.data.empty:
                    self.data = new_data
                    self._process_initial_indexing()
                    return
                
                self._process_data_update(new_data)
                
        except Exception as e:
            st.error(f"API Error: {e}")


    def _row_equal(self, row1, row2):
        """Check if two rows are identical"""
        return row1.to_json() == row2.to_json()


    def _update_loop(self):
        """Background update process"""
        while not self.stop_event.is_set():
            time.sleep(self.interval)
            self._fetch_data()

    
    def get_data(self):
        """Get current data snapshot"""
        with self.lock:
            return self.data.copy()
    
    def stop(self):
        """Stop background updates"""
        self.stop_event.set()
        self.thread.join()



if __name__ == "__main__":
    main()