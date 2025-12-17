import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dateutil import relativedelta as datere
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import os
from warnings import simplefilter
from bs4 import BeautifulSoup
import re
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# ==================== 0. 頁面與字型設定 ====================
st.set_page_config(page_title="Jockey Race", layout="wide")

# --- 自動處理中文字型 (專為 Streamlit Cloud 設計) ---
FONT_URL = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
FONT_FILE = "NotoSansCJKtc-Regular.otf"

@st.cache_resource
def get_chinese_font():
    # 如果字型檔不存在，則下載
    if not os.path.exists(FONT_FILE):
        with st.spinner("正在下載中文字型 (首次運行需要)..."):
            try:
                r = requests.get(FONT_URL)
                with open(FONT_FILE, "wb") as f:
                    f.write(r.content)
            except:
                st.warning("無法下載中文字型，圖表文字可能顯示為方框。")
                return None
    
    # 加入字型管理器
    if os.path.exists(FONT_FILE):
        fm.fontManager.addfont(FONT_FILE)
        # 設定 Matplotlib 全局字型
        plt.rcParams['font.family'] = fm.FontProperties(fname=FONT_FILE).get_name()
    return FONT_FILE

# 初始化字型
get_chinese_font()

st.title("🏇 Jockey Race 賽馬預測 (Streamlit 版)")

# ==================== 1. Session State 初始化 ====================
def init_session_state():
    defaults = {
        'monitoring': False, # 控制是否正在監控
        'reset': False,
        'odds_dict': {},
        'investment_dict': {},
        'overall_investment_dict': {},
        'weird_dict': {},
        'diff_dict': {},
        'race_dict': {},
        'post_time_dict': {},
        'numbered_dict': {},
        'race_dataframes': {},
        'ucb_dict': {},
        'api_called': False,
        'last_update': None,
        'jockey_data': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==================== 2. 數據下載與處理函數 ====================

def get_investment_data():
  url = 'https://info.cld.hkjc.com/graphql/base/'
  headers = {'Content-Type': 'application/json'}

  payload_investment = {
      "operationName": "racing",
      "variables": {
          "date": str(Date),
          "venueCode": place,
          "raceNo": int(race_no),
          "oddsTypes": methodlist
      },
      "query": """
      query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int) {
        raceMeetings(date: $date, venueCode: $venueCode) {
          totalInvestment
          poolInvs: pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) {
            id
            leg {
              number
              races
            }
            status
            sellStatus
            oddsType
            investment
            mergedPoolId
            lastUpdateTime
          }
        }
      }
      """
  }

  response = requests.post(url, headers=headers, json=payload_investment)

  if response.status_code == 200:
      investment_data = response.json()

      # Extracting the investment into different types of oddsType
      investments = {
          "WIN": [],
          "PLA": [],
          "QIN": [],
          "QPL": [],
          "FCT": [],
          "TRI": [],
          "FF": []
      }

      race_meetings = investment_data.get('data', {}).get('raceMeetings', [])
      if race_meetings:
          for meeting in race_meetings:
              pool_invs = meeting.get('poolInvs', [])
              for pool in pool_invs:
                  if place not in ['ST','HV']:
                    id = pool.get('id')
                    if id[8:10] != place:
                      continue                
                  investment = float(pool.get('investment'))
                  investments[pool.get('oddsType')].append(investment)

          #print("Investments:", investments)
      else:
          print("No race meetings found in the response.")

      return investments
  else:
      print(f"Error: {response.status_code}")

def get_odds_data():
  url = 'https://info.cld.hkjc.com/graphql/base/'
  headers = {'Content-Type': 'application/json'}
  payload_odds = {
      "operationName": "racing",
      "variables": {
          "date": str(Date),
          "venueCode": place,
          "raceNo": int(race_no),
          "oddsTypes": methodlist
      },
      "query": """
      query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int) {
        raceMeetings(date: $date, venueCode: $venueCode) {
          pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) {
            id
            status
            sellStatus
            oddsType
            lastUpdateTime
            guarantee
            minTicketCost
            name_en
            name_ch
            leg {
              number
              races
            }
            cWinSelections {
              composite
              name_ch
              name_en
              starters
            }
            oddsNodes {
              combString
              oddsValue
              hotFavourite
              oddsDropValue
              bankerOdds {
                combString
                oddsValue
              }
            }
          }
        }
      }
      """
  }

  response = requests.post(url, headers=headers, json=payload_odds)
  if response.status_code == 200:
      odds_data = response.json()
          # Extracting the oddsValue into different types of oddsType and sorting by combString for QIN and QPL
      # Initialize odds_values with empty lists for each odds type
      odds_values = {
          "WIN": [],
          "PLA": [],
          "QIN": [],
          "QPL": [],
          "FCT": [],
          "TRI": [],
          "FF": []
      }
      
      race_meetings = odds_data.get('data', {}).get('raceMeetings', [])
      for meeting in race_meetings:
          pm_pools = meeting.get('pmPools', [])
          for pool in pm_pools:
              if place not in ['ST', 'HV']:
                  id = pool.get('id')
                  if id and id[8:10] != place:  # Check if id exists before slicing
                      continue
              odds_nodes = pool.get('oddsNodes', [])
              odds_type = pool.get('oddsType')
              odds_values[odds_type] = []
              # Skip if odds_type is invalid or not in odds_values
              if not odds_type or odds_type not in odds_values:
                  continue
              for node in odds_nodes:
                  oddsValue = node.get('oddsValue')
                  # Skip iteration if oddsValue is None, empty, or '---'
                  if oddsValue == 'SCR':
                      oddsValue = np.inf
                  else:
                      try:
                          oddsValue = float(oddsValue)
                      except (ValueError, TypeError):
                          continue  # Skip if oddsValue can't be converted to float
                  # Store data based on odds_type
                  if odds_type in ["QIN", "QPL", "FCT", "TRI", "FF"]:
                      comb_string = node.get('combString')
                      if comb_string:  # Ensure combString exists
                          odds_values[odds_type].append((comb_string, oddsValue))
                  else:
                      odds_values[odds_type].append(oddsValue)
      # Sorting the odds values for specific types by combString in ascending order
      for odds_type in ["QIN", "QPL", "FCT", "TRI", "FF"]:
          odds_values[odds_type].sort(key=lambda x: x[0], reverse=False)
      return odds_values

      #print("WIN Odds Values:", odds_values["WIN"])
      #print("PLA Odds Values:", odds_values["PLA"])
      #print("QIN Odds Values (sorted by combString):", [value for _, value in odds_values["QIN"]])
      #print("QPL Odds Values (sorted by combString):", [value for _, value in odds_values["QPL"]])

  else:
      print(f"Error: {response.status_code}")
def extract_jockey_data(html_content):
    """
    Extracts jockey ranking data from HKJC HTML and returns a Pandas DataFrame.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    ranking_table = soup.select_one('table.table_bd')
    
    if not ranking_table:
        return pd.DataFrame() # Return empty DF if table not found

    jockey_data = []
    headers_chinese = ["騎師", "冠", "亞", "季", "殿", "第五", "總出賽次數", "所贏獎金"]

    # Locate the specific data-containing tbodies
    data_sections = ranking_table.find_all('tbody', class_='f_tac f_fs12')
    
    for tbody in data_sections:
        for row in tbody.find_all('tr'):
            td_elements = row.find_all('td')
            
            if len(td_elements) != len(headers_chinese):
                continue

            row_data = {}
            
            # 1. Extract Jockey Name
            jockey_cell = td_elements[0].find('a')
            row_data["騎師"] = jockey_cell.get_text(strip=True) if jockey_cell else td_elements[0].get_text(strip=True)
            
            # 2. Extract Numbers
            for i in range(1, len(headers_chinese)):
                header = headers_chinese[i]
                raw_value = td_elements[i].get_text(strip=True)
                
                # Clean currency and commas
                clean_value = re.sub(r'[$,]', '', raw_value)
                try:
                    row_data[header] = int(clean_value)
                except ValueError:
                    row_data[header] = 0
            
            jockey_data.append(row_data)

    # Convert the list of dictionaries to a DataFrame immediately
    return pd.DataFrame(jockey_data)


def get_jockey_ranking():
    """
    Fetches the HKJC page and returns the jockey rankings as a DataFrame.
    """
    url = "https://racing.hkjc.com/racing/information/Chinese/Jockey/JockeyRanking.aspx"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # This now returns a DataFrame instead of a list
        df = extract_jockey_data(response.text)
        return df

    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        return pd.DataFrame() # Return empty DF on error

# --- Example of running the function ---
# ranking = get_jockey_ranking()
# if ranking:
#     # Print the top 3 jockeys in a readable format
#     print(json.dumps(ranking[:3], indent=2, ensure_ascii=False))
# else:
#     print("Failed to retrieve or parse ranking data.")
def extract_trainer_data(html_content):
    """
    從香港賽馬會練馬師排名 HTML 內容中提取數據，並返回一個 Pandas DataFrame。
    (Extracts trainer ranking data from HKJC HTML and returns a Pandas DataFrame.)
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 練馬師排名的表格同樣使用 'table_bd' class
    ranking_table = soup.select_one('table.table_bd')
    
    if not ranking_table:
        # 如果找不到表格，返回空的 DataFrame
        return pd.DataFrame() 

    trainer_data = []
    
    # 練馬師排名的欄位標題:
    # Trainer, 1st, 2nd, 3rd, 4th, 5th, Total Runs, Prize Money
    headers_chinese = ["練馬師", "冠", "亞", "季", "殿", "第五", "總出賽次數", "所贏獎金"]

    # 數據同樣位於 class 為 'f_tac f_fs12' 的 tbody 標籤中
    # (現役練馬師 和 其他練馬師)
    data_sections = ranking_table.find_all('tbody', class_='f_tac f_fs12')
    
    for tbody in data_sections:
        for row in tbody.find_all('tr'):
            td_elements = row.find_all('td')
            
            # 確保行中有 8 個數據欄位
            if len(td_elements) != len(headers_chinese):
                continue

            row_data = {}
            
            # 1. 提取練馬師名稱 (位於 <a> 標籤內)
            trainer_cell = td_elements[0].find('a')
            row_data["練馬師"] = trainer_cell.get_text(strip=True) if trainer_cell else td_elements[0].get_text(strip=True)
            
            # 2. 提取數字數據
            for i in range(1, len(headers_chinese)):
                header = headers_chinese[i]
                raw_value = td_elements[i].get_text(strip=True)
                
                # 清理貨幣符號和逗號
                clean_value = re.sub(r'[$,]', '', raw_value)
                try:
                    row_data[header] = int(clean_value)
                except ValueError:
                    row_data[header] = 0
            
            trainer_data.append(row_data)

    # 將字典列表轉換為 DataFrame
    return pd.DataFrame(trainer_data)


def get_trainer_ranking():
    """
    獲取香港賽馬會練馬師排名頁面，並將數據提取為 DataFrame。
    (Fetches the HKJC Trainer Ranking page and returns the data as a DataFrame.)
    """
    url = "https://racing.hkjc.com/racing/information/Chinese/Trainers/TrainerRanking.aspx"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # 檢查 HTTP 請求是否成功
        
        # 將 HTML 內容傳遞給提取函數
        df = extract_trainer_data(response.text)
        return df

    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        return pd.DataFrame() # 請求失敗時返回空的 DataFrame
        
def save_odds_data(time_now,odds):
  for method in methodlist:
      if method in ['WIN', 'PLA']:
        if st.session_state.odds_dict[method].empty:
            # Initialize the DataFrame with the correct number of columns
            st.session_state.odds_dict[method] = pd.DataFrame(columns=np.arange(1, len(odds[method]) + 1))
        st.session_state.odds_dict[method].loc[time_now] = odds[method]
      elif method in ['QIN','QPL',"FCT","TRI","FF"]:
        if odds[method]:
          combination, odds_array = zip(*odds[method])
          if st.session_state.odds_dict[method].empty:
            st.session_state.odds_dict[method] = pd.DataFrame(columns=combination)
            # Set the values with the specified index
          st.session_state.odds_dict[method].loc[time_now] = odds_array
  #st.write(st.session_state.odds_dict)

def save_investment_data(time_now,investment,odds):
  for method in methodlist:
      if method in ['WIN', 'PLA']:
        if st.session_state.investment_dict[method].empty:
            # Initialize the DataFrame with the correct number of columns
            st.session_state.investment_dict[method] = pd.DataFrame(columns=np.arange(1, len(odds[method]) + 1))
        investment_df = [round(investments[method][0]  / 1000 / odd, 2) for odd in odds[method]]
        st.session_state.investment_dict[method].loc[time_now] = investment_df
      elif method in ['QIN','QPL',"FCT","TRI","FF"]:
        if odds[method]:
          combination, odds_array = zip(*odds[method])
          if st.session_state.investment_dict[method].empty:
            st.session_state.investment_dict[method] = pd.DataFrame(columns=combination)
          investment_df = [round(investments[method][0]  / 1000 / odd, 2) for odd in odds_array]
              # Set the values with the specified index
          st.session_state.investment_dict[method].loc[time_now] = investment_df

def investment_combined(time_now,method,df):
  sums = {}
  for col in df.columns:
      # Split the column name to get the numbers
      num1, num2 = col.split(',')
      # Convert to integers
      num1, num2 = int(num1), int(num2)

      # Sum the column values
      col_sum = df[col].sum()

      # Add the sum to the corresponding numbers in the dictionary
      if num1 in sums:
          sums[num1] += col_sum
      else:
          sums[num1] = col_sum

      if num2 in sums:
          sums[num2] += col_sum
      else:
          sums[num2] = col_sum

  # Convert the sums dictionary to a dataframe for better visualization
  sums_df = pd.DataFrame([sums],index = [time_now]) /2
  return sums_df

def get_overall_investment(time_now,dict):
    investment_df = st.session_state.investment_dict
    no_of_horse = len(investment_df['WIN'].columns)
    total_investment_df = pd.DataFrame(index =[time_now], columns=np.arange(1,no_of_horse +1))
    for method in methodlist:
      if method in ['WIN','PLA']:
        st.session_state.overall_investment_dict[method] = st.session_state.overall_investment_dict[method]._append(st.session_state.investment_dict[method].tail(1))
      elif method in ['QIN','QPL']:
        if not investment_df[method].empty:
          st.session_state.overall_investment_dict[method] = st.session_state.overall_investment_dict[method]._append(investment_combined(time_now,method,st.session_state.investment_dict[method].tail(1)))
        else:
          continue

    for horse in range(1,no_of_horse+1):
        total_investment = 0
        for method in methodlist:
            if method in ['WIN', 'PLA']:
                investment = st.session_state.overall_investment_dict[method][horse].values[-1]
            elif method in ['QIN','QPL']:
              if not investment_df[method].empty: 
                investment = st.session_state.overall_investment_dict[method][horse].values[-1]
              else:
                continue
            total_investment += investment
        total_investment_df[horse] = total_investment
    st.session_state.overall_investment_dict['overall'] = st.session_state.overall_investment_dict['overall']._append(total_investment_df)


def weird_data(time_now, investments, odds, methodlist):
    for method in methodlist:
        if st.session_state.investment_dict[method].empty or len(st.session_state.investment_dict[method]) < 2:
            continue
            
        latest_investment = st.session_state.investment_dict[method].tail(1).values
        # Using previous odds for expectation calculation might be safer, but logic follows user code
        last_time_odds_df = st.session_state.odds_dict[method].tail(2).head(1)
        
        if last_time_odds_df.empty: continue
        last_time_odds = last_time_odds_df.values
        
        try:
            pool_total = investments[method][0]
            expected = pool_total / 1000 / last_time_odds
            # Handling infinity/zero division
            expected = np.where(last_time_odds == np.inf, 0, expected)
            
            diff = np.round(latest_investment - expected, 0)
            diff_df = pd.DataFrame(diff, columns=st.session_state.investment_dict[method].columns, index=[time_now])

            if method in ['WIN','PLA']:
                st.session_state.diff_dict[method] = pd.concat([st.session_state.diff_dict.get(method, pd.DataFrame()), diff_df])
            elif method in ['QIN','QPL']:
                combined_diff = investment_combined(time_now, method, diff_df)
                st.session_state.diff_dict[method] = pd.concat([st.session_state.diff_dict.get(method, pd.DataFrame()), combined_diff])
        except Exception as e:
            # st.error(f"Error in weird_data: {e}")
            pass

def weird_data(investments):

  for method in methodlist:
    if st.session_state.investment_dict[method].empty:
      continue
    latest_investment = st.session_state.investment_dict[method].tail(1).values
    last_time_odds = st.session_state.odds_dict[method].tail(2).head(1)
    expected_investment = investments[method][0] / 1000 / last_time_odds
    diff = round(latest_investment - expected_investment,0)
    if method in ['WIN','PLA']:
        st.session_state.diff_dict[method] = st.session_state.diff_dict[method]._append(diff)
    elif method in ['QIN','QPL']:
        st.session_state.diff_dict[method] = st.session_state.diff_dict[method]._append(investment_combined(time_now,method,diff))
    
def change_overall(time_now):
  total_investment = 0
  for method in methodlist:
    total_investment += st.session_state.diff_dict[method].sum(axis=0)
  total_investment_df = pd.DataFrame([total_investment],index = [time_now])
  st.session_state.diff_dict['overall'] = st.session_state.diff_dict['overall']._append(total_investment_df)
# ==================== 3. 繪圖函數 (簡化版) ====================

def print_bubble(race_no, print_list):
    # 確保有數據
    if 'WIN' not in st.session_state.overall_investment_dict or st.session_state.overall_investment_dict['WIN'].empty:
        return

    for method in print_list:
        if method not in ['WIN&QIN', 'PLA&QPL']: continue
        
        try:
            if method == 'WIN&QIN':
                vol_win = st.session_state.overall_investment_dict.get('WIN', pd.DataFrame())
                vol_qin = st.session_state.overall_investment_dict.get('QIN', pd.DataFrame())
                diff_win = st.session_state.diff_dict.get('WIN', pd.DataFrame())
                diff_qin = st.session_state.diff_dict.get('QIN', pd.DataFrame())
                method_name = ['WIN','QIN']
            else:
                vol_win = st.session_state.overall_investment_dict.get('PLA', pd.DataFrame())
                vol_qin = st.session_state.overall_investment_dict.get('QPL', pd.DataFrame())
                diff_win = st.session_state.diff_dict.get('PLA', pd.DataFrame())
                diff_qin = st.session_state.diff_dict.get('QPL', pd.DataFrame())
                method_name = ['PLA','QPL']

            if vol_win.empty or vol_qin.empty or diff_win.empty or diff_qin.empty:
                continue

            total_volume = vol_win.tail(1) + vol_qin.tail(1)
            # Sum last 10 periods for delta
            delta_I = diff_win.tail(10).sum(axis=0) * 10
            delta_Q = diff_qin.tail(10).sum(axis=0) * 10
            
            df = pd.DataFrame({
                'horse': total_volume.columns.astype(str),
                'ΔI': delta_I.values,
                'ΔQ': delta_Q.values,
                '總投注量': total_volume.iloc[0].fillna(0).round(0).astype(int).values
            })
            
            df = df[df['總投注量'] > 0] # Filter out scratched
            if df.empty: continue

            # Normalization for bubble size
            raw_size = df['總投注量']
            bubble_size = 20 + (raw_size - raw_size.min()) / (raw_size.max() - raw_size.min() + 1e-6) * 80
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['ΔI'], y=df['ΔQ'],
                mode='markers+text',
                text=df['horse'],
                textposition="middle center",
                textfont=dict(color="white", size=14, weight="bold"),
                marker=dict(
                    size=bubble_size,
                    sizemode='area',
                    sizeref=2.*bubble_size.max()/(60**2),
                    color=df['ΔI'],
                    colorscale='Bluered_r',
                    reversescale=True,
                    line=dict(width=1, color='white'),
                    opacity=0.8
                ),
                hovertemplate="<b>馬號：%{text}</b><br>總量：%{customdata:,}K<br>Δ%{yaxis.title.text}: %{y:.1f}K<br>Δ%{xaxis.title.text}: %{x:.1f}K",
                customdata=df['總投注量']
            ))

            fig.add_hline(y=0, line_color="lightgrey")
            fig.add_vline(x=0, line_color="lightgrey")
            fig.update_layout(
                title=f"{method} 氣泡圖 (第{race_no}場)",
                xaxis_title=method_name[0],
                yaxis_title=method_name[1],
                height=500,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Bubble Chart Error: {e}")

# ==================== 4. 主介面邏輯 ====================

# --- 輸入區 ---
with st.sidebar:
    st.header("設定")
    Date = st.date_input('日期:', value=datetime.now(timezone(timedelta(hours=8))).date())
    place = st.selectbox('場地:', ['ST', 'HV', 'S1', 'S2'])
    race_no = st.selectbox('場次:', np.arange(1, 12))
    
    st.markdown("---")
    st.subheader("監控選項")
    
    # 監控開關
    monitoring_on = st.toggle("啟動即時監控", value=False)
    
    if st.button("重置所有數據"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- 賽事資料加載 ---
@st.cache_data(ttl=3600)
def fetch_race_card(date_str, venue):
    # 這是一個簡化的 RaceCard 抓取，只抓基本資料以顯示
    # 完整邏輯較長，這裡保留核心概念：抓取馬名與基本資料
    url = 'https://info.cld.hkjc.com/graphql/base/'
    headers = {'Content-Type': 'application/json'}
    payload = {
        "operationName": "raceMeetings",
        "variables": {"date": date_str, "venueCode": venue},
        "query": """
      fragment raceFragment on Race {
        id
        no
        status
        raceName_en
        raceName_ch
        postTime
        country_en
        country_ch
        distance
        wageringFieldSize
        go_en
        go_ch
        ratingType
        raceTrack {
          description_en
          description_ch
        }
        raceCourse {
          description_en
          description_ch
          displayCode
        }
        claCode
        raceClass_en
        raceClass_ch
        judgeSigns {
          value_en
        }
      }
  
      fragment racingBlockFragment on RaceMeeting {
        jpEsts: pmPools(
          oddsTypes: [TCE, TRI, FF, QTT, DT, TT, SixUP]
          filters: ["jackpot", "estimatedDividend"]
        ) {
          leg {
            number
            races
          }
          oddsType
          jackpot
          estimatedDividend
          mergedPoolId
        }
        poolInvs: pmPools(
          oddsTypes: [WIN, PLA, QIN, QPL, CWA, CWB, CWC, IWN, FCT, TCE, TRI, FF, QTT, DBL, TBL, DT, TT, SixUP]
        ) {
          id
          leg {
            races
          }
        }
        penetrometerReadings(filters: ["first"]) {
          reading
          readingTime
        }
        hammerReadings(filters: ["first"]) {
          reading
          readingTime
        }
        changeHistories(filters: ["top3"]) {
          type
          time
          raceNo
          runnerNo
          horseName_ch
          horseName_en
          jockeyName_ch
          jockeyName_en
          scratchHorseName_ch
          scratchHorseName_en
          handicapWeight
          scrResvIndicator
        }
      }
  
      query raceMeetings($date: String, $venueCode: String) {
        timeOffset {
          rc
        }
        activeMeetings: raceMeetings {
          id
          venueCode
          date
          status
          races {
            no
            postTime
            status
            wageringFieldSize
          }
        }
        raceMeetings(date: $date, venueCode: $venueCode) {
          id
          status
          venueCode
          date
          totalNumberOfRace
          currentNumberOfRace
          dateOfWeek
          meetingType
          totalInvestment
          country {
            code
            namech
            nameen
            seq
          }
          races {
            ...raceFragment
            runners {
              id
              no
              standbyNo
              status
              name_ch
              name_en
              horse {
                id
                code
              }
              color
              barrierDrawNumber
              handicapWeight
              currentWeight
              currentRating
              internationalRating
              gearInfo
              racingColorFileName
              allowance
              trainerPreference
              last6run
              saddleClothNo
              trumpCard
              priority
              finalPosition
              deadHeat
              winOdds
              jockey {
                code
                name_en
                name_ch
              }
              trainer {
                code
                name_en
                name_ch
              }
            }
          }
          obSt: pmPools(oddsTypes: [WIN, PLA]) {
            leg {
              races
            }
            oddsType
            comingleStatus
          }
          poolInvs: pmPools(
            oddsTypes: [WIN, PLA, QIN, QPL, CWA, CWB, CWC, IWN, FCT, TCE, TRI, FF, QTT, DBL, TBL, DT, TT, SixUP]
          ) {
            id
            leg {
              number
              races
            }
            status
            sellStatus
            oddsType
            investment
            mergedPoolId
            lastUpdateTime
          }
          ...racingBlockFragment
          pmPools(oddsTypes: []) {
            id
          }
          jkcInstNo: foPools(oddsTypes: [JKC], filters: ["top"]) {
            instNo
          }
          tncInstNo: foPools(oddsTypes: [TNC], filters: ["top"]) {
            instNo
          }
        }
      }
      """
  }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            races = data.get('data', {}).get('raceMeetings', [])
            race_info = {}
            for meeting in races:
                for race in meeting.get('races', []):
                    r_no = race['no']
                    runners = race.get('runners', [])
                    
                    # 關鍵修改：過濾後備馬匹 (standbyNo 為空字串或 None)
                    filtered_runners = [r for r in runners if not r.get('standbyNo')]

                    data_list = []
                    for r in filtered_runners:
                        
                        # --- 關鍵修正：將字串評分轉換為整數 ---
                        try:
                            # 讀取字串並轉換為整數 (int("059") -> 59)
                            rating_val = int(r.get('currentRating', '0'))
                        except (ValueError, TypeError):
                            rating_val = 0
                            
                        # 排位和負磅也同樣進行穩健的數字轉換
                        try:
                            draw_val = int(r.get('barrierDrawNumber', '0'))
                        except (ValueError, TypeError):
                            draw_val = 0

                        try:
                            weight_val = int(r.get('handicapWeight', '0'))
                        except (ValueError, TypeError):
                            weight_val = 0
                        
                        
                        data_list.append({
                            "馬號": r['no'],
                            "馬名": r['name_ch'],
                            "騎師": r['jockey']['name_ch'] if r['jockey'] else '',
                            "練馬師": r['trainer']['name_ch'] if r['trainer'] else '',
                            "近績": r.get('last6run', ''),
                            
                            # 使用轉換後的數值
                            "評分": rating_val,
                            "排位": draw_val,
                            "負磅": weight_val
                        })

                    df = pd.DataFrame(data_list)
                    if not df.empty:
                        # 將馬號轉換為數字並排序，確保順序正確
                        df['馬號_int'] = pd.to_numeric(df['馬號'], errors='coerce')
                        df = df.sort_values("馬號_int").drop(columns=['馬號_int']).set_index("馬號")
                    
                    # Post Time
                    pt_str = race.get("postTime")
                    pt = datetime.fromisoformat(pt_str) if pt_str else None
                    
                    race_info[r_no] = {"df": df, "post_time": pt}
            return race_info
    except Exception as e:
        print(e)
    return {}
    
def parse_form_score(last6run_str):
    """
    將 '1/2/4/11/2' 這樣的字串轉換為實力分數 (0-100)
    名次越小分數越高。
    """
    if not last6run_str or not isinstance(last6run_str, str):
        return 50 # 預設值
    
    try:
        # 提取最近 3 場名次 (越近期的權重越高)
        runs = []
        parts = last6run_str.split('/')
        for p in parts:
            # 處理像 '12DH' 或 'PU' 這樣的異常值
            clean_p = ''.join(filter(str.isdigit, p))
            if clean_p:
                runs.append(int(clean_p))
        
        if not runs:
            return 50
            
        # 取最近 4 場
        recent_runs = runs[-4:] 
        
        # 計算平均名次 (加權：越近期的比賽權重越重)
        weights = [1, 1.2, 1.5, 2.0] # 權重
        weighted_sum = 0
        total_weight = 0
        
        # 對齊權重與場次
        actual_weights = weights[-len(recent_runs):]
        
        for r, w in zip(recent_runs, actual_weights):
            weighted_sum += r * w
            total_weight += w
            
        avg_rank = weighted_sum / total_weight
        
        # 轉換為分數 (1名=100分, 14名=0分)
        # 公式: 100 - (名次 - 1) * (100 / 13)
        score = 100 - (avg_rank - 1) * 7.7
        return max(0, min(100, score))
        
    except Exception:
        return 50

def calculate_jockey_score(jockey_name, ranking_df):
    """
    根據騎師的排名數據計算其專業分數。
    分數基於當前賽季的勝率，並使用對數平滑化來減少極端值影響。
    """
    if ranking_df.empty:
        return 50 # 無數據給予平均分

    jockey_row = ranking_df[ranking_df['騎師'] == jockey_name]
    
    if jockey_row.empty:
        return 50 # 找不到該騎師，給予平均分

    wins = jockey_row['冠'].iloc[0]
    runs = jockey_row['總出賽次數'].iloc[0]
    
    if runs == 0:
        return 50 # 未出賽，給予平均分
    
    win_rate = wins / runs
    
    # 標準化和加權：將勝率 (0-1) 轉換為 1-100 分。
    # 使用 log 函數對勝場數進行平滑處理，避免極少數出賽次數對勝率分數的誇大。
    # log(wins + 1) * win_rate
    score = (win_rate * 100) * (log(wins + 1) / log(ranking_df['冠'].max() + 1))
    
    # 確保分數在合理範圍內，例如最高 100
    return min(score, 100)


def calculate_trainer_score(trainer_name, ranking_df):
    """
    根據練馬師的排名數據計算其專業分數。
    邏輯與騎師分數相似，但針對練馬師欄位。
    """
    if ranking_df.empty:
        return 50

    trainer_row = ranking_df[ranking_df['練馬師'] == trainer_name]
    
    if trainer_row.empty:
        return 50

    wins = trainer_row['冠'].iloc[0]
    runs = trainer_row['總出賽次數'].iloc[0]
    
    if runs == 0:
        return 50
    
    win_rate = wins / runs
    
    # 標準化和加權
    score = (win_rate * 100) * (log(wins + 1) / log(ranking_df['冠'].max() + 1))
    
    return min(score, 100)
def calculate_smart_score(race_no):
    """
    計算單場賽事的綜合評分，並將所有中間結果整合到單一 df。
    """
    
    # ----------------------------------------------------
    # I. 數據準備與初始 df 建立
    # ----------------------------------------------------
    
    # 1. 獲取最新賠率 (Odds)
    if 'WIN' not in st.session_state.odds_dict or st.session_state.odds_dict['WIN'].empty:
        return pd.DataFrame()
        
    latest_odds = st.session_state.odds_dict['WIN'].tail(1).T
    latest_odds.columns = ['Odds']
    
    # 2. 獲取資金流向 (MoneyFlow)
    if 'WIN' in st.session_state.diff_dict and not st.session_state.diff_dict['WIN'].empty:
        money_flow = st.session_state.diff_dict['WIN'].tail(10).sum().to_frame(name='MoneyFlow')
    else:
        money_flow = pd.DataFrame(0, index=latest_odds.index, columns=['MoneyFlow'])
        
    # 3. 建立基礎 df (包含動態數據)
    df = pd.concat([latest_odds, money_flow], axis=1)
    
    # 4. 獲取靜態數據
    if race_no not in st.session_state.race_dataframes:
        return pd.DataFrame()
        
    # 我們只需要 '馬號' 和計算分數所需的欄位
    static_df = st.session_state.race_dataframes[race_no].copy()
    
    # ----------------------------------------------------
    # II. 索引標準化 (確保合併成功)
    # ----------------------------------------------------
    
    # 確保 static_df 以 '馬號' 作為索引
    if static_df.index.name != '馬號':
        static_df = static_df.reset_index().set_index('馬號')
        
    # **關鍵步驟：強制將兩個 DataFrame 的索引類型統一為字串**
    try:
        df.index = df.index.astype(str)
        static_df.index = static_df.index.astype(str)
    except Exception as e:
        print(f"索引轉換錯誤: {e}")
        return pd.DataFrame()
        
    # ----------------------------------------------------
    # III. 靜態數據分數計算 (在 static_df 上計算)
    # ----------------------------------------------------
    
    # 檢查並補齊必要的欄位
    required_cols = ['近績', '評分', '排位'] # 只需要計算所需欄位
    for col in required_cols:
        if col not in static_df.columns:
            static_df[col] = 0
            
    # 1. 狀態分數 (Form Score) - 權重 40%
    static_df['FormScore'] = static_df['近績'].apply(parse_form_score)
    
    # 2. 配搭/專業分數 (Synergy Score) - 權重 30% (佔位)
    static_df['SynergyScore'] = 70
    
    # 3. 適應性分數 (Draw Score) - 權重 20%
    static_df['排位_int'] = pd.to_numeric(static_df['排位'], errors='coerce').fillna(99)
    static_df['DrawScore'] = 100 - (static_df['排位_int'] - 1) * (100 / 13) 
    
    # 4. 負擔分數 (Rating Score) - 權重 10%
    static_df['Rating_int'] = pd.to_numeric(static_df['評分'], errors='coerce').fillna(0)
    max_rating = static_df['Rating_int'].replace(0, np.nan).max() # 避免 max_rating 為 0
    
    if pd.isna(max_rating):
        static_df['RatingDiffScore'] = 50
    else:
        static_df['RatingDiffScore'] = (static_df['Rating_int'] / max_rating) * 100 
    
    # 最終靜態加權公式
    static_df['TotalFormScore'] = (static_df['FormScore'] * 0.4) + \
                                  (static_df['SynergyScore'] * 0.3) + \
                                  (static_df['DrawScore'] * 0.2) + \
                                  (static_df['RatingDiffScore'] * 0.1)
    
    # ----------------------------------------------------
    # IV. 使用 join/merge 將靜態分數整合到 df (達成單一 df 目的)
    # ----------------------------------------------------
    
    # 只取出計算好的分數欄位
    static_scores = static_df[['馬名','TotalFormScore', 'FormScore', 'SynergyScore', 'DrawScore', 'RatingDiffScore']]
    
    # 使用 join 進行合併：左連接，以 df 的馬號為準。
    # 由於索引已統一為字串，join 將正確地按馬號匹配。
    df = df.join(static_scores, how='left')
    
    # 如果有馬匹在靜態數據中找不到 (例如 TotalFormScore 為 NaN)，則填入預設值
    df['TotalFormScore'] = df['TotalFormScore'].fillna(50) 
    
    # ----------------------------------------------------
    # V. 在單一 df 上計算最終綜合得分 (TotalScore)
    # ----------------------------------------------------
    
    # A. 資金分數 (MoneyScore)
    min_flow = df['MoneyFlow'].min()
    max_flow = df['MoneyFlow'].max()
    
    # 避免 MoneyFlow 都是 0 時除以 0
    if max_flow != min_flow:
        df['MoneyScore'] = (df['MoneyFlow'] - min_flow) / (max_flow - min_flow) * 100
    else:
        df['MoneyScore'] = 50
        
    # B. 價值分數 (ValueScore: 隱含勝率/熱度)
    # 避免 Odds 為 0 或 NaN 時除以 0
    df['ValueScore'] = np.where(df['Odds'].replace(0, np.nan).isna(), 0, (1 / df['Odds']) * 100)
    
    # C. 最終加權公式 (實力 30% + 資金流向 50% + 賠率熱度 20%)
    df['TotalScore'] = (df['TotalFormScore'] * 0.3) + \
                       (df['MoneyScore'] * 0.5) + \
                       (df['ValueScore'] * 0.2)
                            
    return df.sort_values('TotalScore', ascending=False)
    
def calculate_smart_score_static(race_no):
    """
    核心預測算法（靜態版）：專為比賽前一日，缺乏賠率和資金流數據時設計。
    權重：狀態 (40%) + 配搭 (30%) + 適應性 (20%) + 負擔 (10%)
    """
    if race_no not in st.session_state.race_dataframes:
        return pd.DataFrame()
    
    static_df = st.session_state.race_dataframes[race_no].copy()
    
    # 確保所有馬匹都有一個馬號索引
    if static_df.index.name != '馬號':
        static_df = static_df.reset_index().set_index('馬號')

    # 檢查關鍵欄位是否存在 (如果沒有，需要先在 fetch_race_card 中獲取)
    required_cols = ['近績', '評分', '排位', '騎師', '練馬師']
    for col in required_cols:
        if col not in static_df.columns:
            # 這是為了兼容，但建議您去 fetch_race_card 補齊這些欄位
            static_df[col] = 0 
            
    # 1. 狀態分數 (Form Score) - 權重 40%
    # 使用原有的 parse_form_score
    static_df['FormScore'] = static_df['近績'].apply(parse_form_score)
    
    # 2. 騎師分數 (Jockey Score) - 權重 15% (取代部分 Synergy)
    jockey_df = st.session_state.get('jockey_ranking_df', pd.DataFrame())
    trainer_df = st.session_state.get('trainer_ranking_df', pd.DataFrame())
    if '騎師' in static_df.columns:
        static_df['JockeyScore'] = static_df['騎師'].apply(
            lambda x: calculate_jockey_score(x, jockey_df)
        )
    else:
        static_df['JockeyScore'] = 50.0
        
    # 3. 練馬師分數 (Trainer Score) - 權重 15% (取代部分 Synergy)
    if '練馬師' in static_df.columns:
        static_df['TrainerScore'] = static_df['練馬師'].apply(
            lambda x: calculate_trainer_score(x, trainer_df)
        )
    else:
        static_df['TrainerScore'] = 50.0
    
    # 3. 適應性分數 (Adaptability Score) - 權重 20%
    # 排位（檔位）：在該場地/距離下，外檔或內檔表現如何？
    # 假設：通常內檔 (1-4) 較好，中檔 (5-8) 次之，外檔 (9+) 較差
    
    static_df['排位_int'] = pd.to_numeric(static_df['排位'], errors='coerce').fillna(99)
    static_df['DrawScore'] = 100 - (static_df['排位_int'] - 1) * (100 / 13) # 1號檔 100分，14號檔 0分
    
    # 4. 負擔分數 (Burden Score) - 權重 10%
    # 評分與負磅的關係：評分越高負磅越重，負擔越大
    # 簡化：評分最高的馬匹，給予負擔分數較低（因為大家都看好它，但它要負重）
    static_df['Rating_int'] = pd.to_numeric(static_df['評分'], errors='coerce').fillna(0)
    max_rating = static_df['Rating_int'].max()
    
    # 評分差異分數 (相對分數)：評分接近最高分者得分較高
    static_df['RatingDiffScore'] = (static_df['Rating_int'] / max_rating) * 100
    
    # --- 最終加權公式 (完全基於靜態數據) ---
    df = static_df.copy()
    
    df['TotalScore'] = (df['FormScore'] * 0.40) + \
                       (df['JockeyScore'] * 0.15) + \
                       (df['TrainerScore'] * 0.15) + \
                       (df['DrawScore'] * 0.20) + \
                       (df['RatingDiffScore'] * 0.10)
                       
    # 清理並輸出
    output_cols = ['馬名', 'FormScore', 'JockeyScore', 'TrainerScore', 
                   'DrawScore', 'RatingDiffScore', 'TotalScore']
    
    # 只選取存在的欄位
    final_cols = [col for col in output_cols if col in df.columns]

    df = df[final_cols].sort_values('TotalScore', ascending=False)
    
    return df
# 嘗試加載 Race Card
date_str = str(Date)
if not st.session_state.api_called:
    with st.spinner("載入賽事資料中..."):
        race_card_data = fetch_race_card(date_str, place)
        if race_card_data:
            st.session_state.race_dataframes = {k: v['df'] for k,v in race_card_data.items()}
            st.session_state.post_time_dict = {k: v['post_time'] for k,v in race_card_data.items()}
            st.session_state.api_called = True

# --- 顯示賽事資訊 ---
if race_no in st.session_state.race_dataframes:
    pt = st.session_state.post_time_dict.get(race_no)
    pt_str = pt.strftime("%H:%M") if pt else "--:--"
    st.info(f"📍 {place} 第 {race_no} 場 | 🕒 開跑: {pt_str}")
    with st.expander("查看排位表", expanded=False):
        st.dataframe(st.session_state.race_dataframes[race_no], use_container_width=True)
else:
    st.warning("找不到此場次資料，請確認日期與場地。")

# ==================== 5. 監控循環邏輯 ====================

methodlist = ['WIN', 'PLA', 'QIN', 'QPL'] # 簡化預設
print_list = ['WIN&QIN', 'PLA&QPL']
for method in methodlist:
    # 確保 odds_dict, investment_dict, overall_investment_dict, diff_dict 都有 WIN/PLA/QIN/QPL 鍵
    st.session_state.odds_dict.setdefault(method, pd.DataFrame())
    st.session_state.investment_dict.setdefault(method, pd.DataFrame())
    st.session_state.overall_investment_dict.setdefault(method, pd.DataFrame())
    st.session_state.diff_dict.setdefault(method, pd.DataFrame())
    
# 確保 overall 鍵存在於整體投注量和差異字典中
st.session_state.overall_investment_dict.setdefault('overall', pd.DataFrame())
st.session_state.diff_dict.setdefault('overall', pd.DataFrame())
if monitoring_on:
    # --- 實時監控模式 (比賽當日) ---
    st.markdown("### 🟢 實時監控與資金流預測中...")
    placeholder = st.empty()
    
    time_now = datetime.now()
    time_str = time_now.strftime('%H:%M:%S')
    
    # 1. 抓取數據 (這裡需要您的實際抓取邏輯)
    odds = get_odds_data()
    investments = get_investment_data()
    
    if odds and investments:
        with st.spinner(f"更新數據中 ({time_str})..."):
            # 2. 處理數據
            # 這裡需要您的 
            save_odds_data(time_now,odds)
            save_investment_data(time_now,investments,odds)
            get_overall_investment(time_now,investments)
            weird_data(investments)
            change_overall(time_now)
            # 由於篇幅限制，假設已運行
            st.session_state.last_update = time_now

    # 3. 顯示結果
    with placeholder.container():
        st.metric("最後更新", st.session_state.last_update.strftime('%H:%M:%S') if st.session_state.last_update else "N/A")
        
        # A. 氣泡圖 (資金流向視覺化)
        print_bubble(race_no, print_list)
        
        # B. 實時預測排名
        st.markdown("### 🤖 實時資金流綜合預測排名")
        prediction_df = calculate_smart_score(race_no)
        
        if not prediction_df.empty:
            display_df = prediction_df.copy()
            display_df = display_df[['馬名','Odds', 'MoneyFlow', 'TotalFormScore', 'TotalScore']]
            display_df.columns = ['馬名','當前賠率', '近期資金流(K)', '近績評分', '🔥綜合推薦分']
            display_df['當前賠率'] = display_df['當前賠率'].apply(lambda x: f"{x:.1f}")
            display_df['近期資金流(K)'] = display_df['近期資金流(K)'].apply(lambda x: f"{x:.1f}")
            display_df['近績評分'] = display_df['近績評分'].astype(float).round(0).astype('Int64')
            display_df['🔥綜合推薦分'] = display_df['🔥綜合推薦分'].astype(float).round(0).astype('Int64')
            
            def highlight_top_realtime(row):
                # 【關鍵修正：檢查 NaN】
                # 如果 '🔥綜合推薦分' 是 NaN (空值)，則不進行高亮，返回空字串列表
                if pd.isna(row['🔥綜合推薦分']):
                    return [''] * len(row)
        
                # 這裡假設 prediction_df 已經排序，並取其最大值作為比較基礎
                # 由於 TotalScore 來自於計算，它應該是 float 或 NaN。
                top_score = prediction_df['TotalScore'].max()
                
                # 【修正：使用 row 的值】
                # row['🔥綜合推薦分'] 已經是 float 或 Int64 類型，可以直接比較，不需要再次 float() 轉換。
                current_score = row['🔥綜合推薦分']
                
                # 確保 top_score 不是 NaN，避免與 NaN 比較
                if pd.isna(top_score):
                    return [''] * len(row)
        
                # 比較邏輯
                # 這裡的比較值應根據您的業務邏輯定義 (例如: 總分最高 vs 總分第二高)
                # 由於 prediction_df 應該是排序好的，top_score 應該是 prediction_df['TotalScore'].iloc[0] (最高分)
                
                # 為了安全，我們使用 max()
                # 假設您的邏輯是與最高分和第二高分比較：
                
                # 1. 找出最高分
                top_score = prediction_df['TotalScore'].max()
                # 2. 找出第二高分 (如果只有一匹馬，這個會是 NaN 或與最高分相同)
                second_top_score = prediction_df['TotalScore'].nlargest(2).iloc[-1] if len(prediction_df) >= 2 else top_score
        
                # 紅色高亮：最高分
                if current_score == top_score:
                    return ['background-color: #ffcccc'] * len(row)
                # 黃色高亮：第二高分 (或與最高分接近的分數)
                elif current_score == second_top_score:
                     return ['background-color: #ffffcc'] * len(row)
                else:
                    return [''] * len(row)
        
            # 應用高亮函數
            st.dataframe(display_df.style.apply(highlight_top_realtime, axis=1), use_container_width=True)
            st.info(f"💡 AI 實時建議：目前綜合數據最強的是 **{display_df.index[0]}號馬** (基於資金流、賠率和近績)。")

    # 4. 自動刷新機制
    time.sleep(15) 
    st.rerun()     

# --- 在主介面邏輯 (第 350 行左右) 增加一個賽前預測模式 ---
if not monitoring_on: # 只有當實時監控關閉時，才提供靜態預測
    
    st.markdown("### 🔍 賽前靜態預測分析")
    st.info("由於缺乏實時賠率和資金流數據，本分析完全基於馬匹、騎師和場地等靜態資訊。")
    
    # 執行靜態預測
    static_prediction_df = calculate_smart_score_static(race_no)
    jockey_ranking = get_jockey_ranking()
    st.write(jockey_ranking)
    trainer_ranking = get_trainer_ranking()
    st.write(trainer_ranking)
    if not static_prediction_df.empty:
        # 整理顯示格式
        display_df = static_prediction_df.copy()
        display_df = display_df[['馬名', 'FormScore', 'JockeyScore', 'TrainerScore', 
                   'DrawScore', 'RatingDiffScore', 'TotalScore']]
        display_df.columns = ['馬名','近績狀態分','騎師分','練馬師分', '檔位優勢分', '評分負擔分', '🏆 靜態預測分']
        
        # 格式化
        display_df['近績狀態分'] = display_df['近績狀態分'].astype(int)
        display_df['騎師分'] = display_df['騎師分'].astype(int)
        display_df['練馬師分'] = display_df['練馬師分'].astype(int)
        display_df['檔位優勢分'] = display_df['檔位優勢分'].astype(int)
        display_df['評分負擔分'] = display_df['評分負擔分'].astype(int)
        display_df['🏆 靜態預測分'] = display_df['🏆 靜態預測分'].apply(lambda x: f"{x:.1f}")

        # 高亮處理...
        # （與前一回答中的高亮邏輯相同）
        def highlight_top_static(row):
            top_score = static_prediction_df['TotalScore'].max()
            current_score = row['TotalScore'] if 'TotalScore' in row else 0
            
            if current_score >= top_score:
                return ['background-color: #ffcccc'] * len(row)
            elif current_score >= static_prediction_df['TotalScore'].nlargest(3).iloc[-1]:
                return ['background-color: #ffffcc'] * len(row)
            else:
                return [''] * len(row)

        st.dataframe(display_df.style.apply(highlight_top_static, axis=1), use_container_width=True)
        
        top_horse_static = display_df.index[0]
        st.success(f"🏅 賽前靜態預測：**{top_horse_static}號馬** 具有最佳的**近績與排位**組合優勢。")
