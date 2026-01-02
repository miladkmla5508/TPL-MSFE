import numpy as np
import pandas as pd

# =============================================================================
# Helper function to create threshold arrays
# =============================================================================
def make_threshold(value, length=252):
    """Create a constant threshold array"""
    return np.full(length, value)

# =============================================================================
# Get the last 252 rows efficiently
# =============================================================================
df_last = df.iloc[-252:].copy()

# =============================================================================
# BOLLINGER BANDS FILL REGIONS
# =============================================================================
BB_fill_up = dict(
    y1=df_last['BB_P_8'].values, 
    y2=make_threshold(0.3), 
    where=df_last['BB_P_8'].values <= 0.3, 
    alpha=0.5, 
    color='#ff7770'
)

BB_fill_down = dict(
    y1=df_last['BB_P_8'].values, 
    y2=make_threshold(0.7), 
    where=df_last['BB_P_8'].values > 0.7, 
    alpha=0.5, 
    color='#5fd448'
)

# =============================================================================
# KELTNER CHANNEL FILL REGIONS
# =============================================================================
KC_fill_up = dict(
    y1=df_last['KC_P_10'].values, 
    y2=make_threshold(0), 
    where=df_last['KC_P_10'].values <= 0, 
    alpha=0.5, 
    color='#ff7770'
)

KC_fill_down = dict(
    y1=df_last['KC_P_10'].values, 
    y2=make_threshold(1), 
    where=df_last['KC_P_10'].values > 1, 
    alpha=0.5, 
    color='#5fd448'
)

# =============================================================================
# AROON FILL REGIONS
# =============================================================================
ARU_fill_up = dict(
    y1=df_last['AROON_9'].values, 
    y2=make_threshold(-50), 
    where=df_last['AROON_9'].values <= -50, 
    alpha=0.5, 
    color='#ff7770'
)

ARU_fill_down = dict(
    y1=df_last['AROON_9'].values, 
    y2=make_threshold(50), 
    where=df_last['AROON_9'].values > 50, 
    alpha=0.5, 
    color='#5fd448'
)

AR_fill_up = dict(
    y1=df_last['AROON_9'].values, 
    y2=df_last['AROON_UP_7'].values, 
    where=df_last['AROON_9'].values <= df_last['AROON_UP_7'].values, 
    alpha=0.5, 
    color='#5fd448'
)

AR_fill_down = dict(
    y1=df_last['AROON_9'].values, 
    y2=df_last['AROON_UP_7'].values, 
    where=df_last['AROON_9'].values > df_last['AROON_UP_7'].values, 
    alpha=0.5, 
    color='#ff7770'
)

# =============================================================================
# ADX POSITIVE FILL REGIONS
# =============================================================================
ADXP_fill_up = dict(
    y1=df_last['ADX_POS_9'].values, 
    y2=make_threshold(20), 
    where=df_last['ADX_POS_9'].values <= 20, 
    alpha=0.5, 
    color='#ff7770'
)

ADXP_fill_down = dict(
    y1=df_last['ADX_POS_9'].values, 
    y2=make_threshold(30), 
    where=df_last['ADX_POS_9'].values > 30, 
    alpha=0.5, 
    color='#5fd448'
)

# =============================================================================
# DONCHIAN CHANNEL FILL REGIONS
# =============================================================================
DC_fill_up = dict(
    y1=df_last['DC_P_18'].values, 
    y2=make_threshold(0.3), 
    where=df_last['DC_P_18'].values <= 0.3, 
    alpha=0.5, 
    color='#ff7770'
)

DC_fill_down = dict(
    y1=df_last['DC_P_18'].values, 
    y2=make_threshold(0.7), 
    where=df_last['DC_P_18'].values > 0.7, 
    alpha=0.5, 
    color='#5fd448'
)

# Note: These fill regions compare the same column to itself (always false)
# Keeping them as in original but they won't show any fill
DCP_fill_up = dict(
    y1=df_last['DC_P_18'].values, 
    y2=df_last['DC_P_18'].values, 
    where=df_last['DC_P_18'].values <= df_last['DC_P_18'].values, 
    alpha=0.5, 
    color='#ff7770'
)

DCP_fill_down = dict(
    y1=df_last['DC_P_18'].values, 
    y2=df_last['DC_P_18'].values, 
    where=df_last['DC_P_18'].values > df_last['DC_P_18'].values, 
    alpha=0.5, 
    color='#5fd448'
)

# =============================================================================
# ULTIMATE OSCILLATOR FILL REGIONS
# =============================================================================
UO_fill_up = dict(
    y1=df_last['ULTOSC_9'].values, 
    y2=make_threshold(40), 
    where=df_last['ULTOSC_9'].values <= 40, 
    alpha=0.5, 
    color='#ff7770'
)

UO_fill_down = dict(
    y1=df_last['ULTOSC_9'].values, 
    y2=make_threshold(50), 
    where=df_last['ULTOSC_9'].values > 50, 
    alpha=0.5, 
    color='#5fd448'
)

# =============================================================================
# MACD FILL REGIONS
# =============================================================================
MACD_fill_up = dict(
    y1=df_last['MACD_DIFF_9'].values, 
    y2=make_threshold(-0.3), 
    where=df_last['MACD_DIFF_9'].values <= -0.3, 
    alpha=0.5, 
    color='#ff7770'
)

MACD_fill_down = dict(
    y1=df_last['MACD_DIFF_9'].values, 
    y2=make_threshold(0.9), 
    where=df_last['MACD_DIFF_9'].values > 0.9, 
    alpha=0.5, 
    color='#5fd448'
)

# =============================================================================
# STOCHASTIC OSCILLATOR FILL REGIONS
# =============================================================================
SLOW_fill_up = dict(
    y1=df_last['SLOWD_6'].values / 100, 
    y2=make_threshold(0.3), 
    where=df_last['SLOWD_6'].values / 100 <= 0.3, 
    alpha=0.5, 
    color='#ff7770'
)

SLOW_fill_down = dict(
    y1=df_last['SLOWD_6'].values / 100, 
    y2=make_threshold(0.7), 
    where=df_last['SLOWD_6'].values / 100 > 0.7, 
    alpha=0.5, 
    color='#5fd448'
)

# =============================================================================
# CCI FILL REGIONS
# =============================================================================
CCI_fill_up = dict(
    y1=df_last['CCI_6'].values, 
    y2=make_threshold(100), 
    where=df_last['CCI_6'].values >= 100, 
    alpha=0.5, 
    color='#5fd448'
)

CCI_fill_down = dict(
    y1=df_last['CCI_6'].values, 
    y2=make_threshold(-100), 
    where=df_last['CCI_6'].values < -100, 
    alpha=0.5, 
    color='#ff7770'
)

# =============================================================================
# WILLIAMS %R FILL REGIONS
# =============================================================================
WILLIAM_fill_up = dict(
    y1=df_last['WILLIAM_7'].values, 
    y2=make_threshold(-20), 
    where=df_last['WILLIAM_7'].values >= -20, 
    alpha=0.5, 
    color='#5fd448'
)

WILLIAM_fill_down = dict(
    y1=df_last['WILLIAM_7'].values, 
    y2=make_threshold(-80), 
    where=df_last['WILLIAM_7'].values < -80, 
    alpha=0.5, 
    color='#ff7770'
)

# =============================================================================
# TSI FILL REGIONS
# =============================================================================
TSI_fill_up = dict(
    y1=df_last['TSI_6'].values, 
    y2=make_threshold(30), 
    where=df_last['TSI_6'].values >= 30, 
    alpha=0.5, 
    color='#5fd448'
)

TSI_fill_down = dict(
    y1=df_last['TSI_6'].values, 
    y2=make_threshold(-30), 
    where=df_last['TSI_6'].values < -30, 
    alpha=0.5, 
    color='#ff7770'
)

# =============================================================================
# RSI FILL REGIONS
# =============================================================================
RSI_fill_up = dict(
    y1=df_last['RSI_6'].values, 
    y2=make_threshold(60), 
    where=df_last['RSI_6'].values >= 60, 
    alpha=0.5, 
    color='#5fd448'
)

RSI_fill_down = dict(
    y1=df_last['RSI_6'].values, 
    y2=make_threshold(40), 
    where=df_last['RSI_6'].values < 40, 
    alpha=0.5, 
    color='#ff7770'
)

# =============================================================================
# CMO FILL REGIONS
# =============================================================================
CMO_fill_up = dict(
    y1=df_last['CMO_16'].values, 
    y2=make_threshold(20), 
    where=df_last['CMO_16'].values >= 20, 
    alpha=0.5, 
    color='#5fd448'
)

CMO_fill_down = dict(
    y1=df_last['CMO_16'].values, 
    y2=make_threshold(-20), 
    where=df_last['CMO_16'].values < -20, 
    alpha=0.5, 
    color='#ff7770'
)

# =============================================================================
# CREATE LABEL1 COLUMN WITH DIFFERENCES
# =============================================================================
# Calculate differences using vectorized operations
list_test = y_test_window_XGB  # This varible is the precidtion results of the LR model
diff_array = np.diff(list_test, prepend=list_test[0])

# Create LABEL1 column properly
df['LABEL1'] = df['LABEL'].copy()
df.loc[df.index[-252:], 'LABEL1'] = diff_array[-252:]

print("Fill regions created successfully!")
print(f"Total fill region definitions: 24")








import mplfinance as fplt

#technical_indicators1 = fplt.make_addplot(df["Close"][0:100],linestyle='solid',color='black', width=1)

def detect_intraday_price_diff(data):
    up_markers = []
    down_markers = []
    for index, row in data.iterrows():
        if row['LABEL1'] == 1:
          up_markers.append(row['Low']-1)
          down_markers.append(np.nan)
        elif row['LABEL1'] == -1 :
          down_markers.append(row['High']+1)
          up_markers.append(np.nan)
        else:
          up_markers.append(np.nan)
          down_markers.append(np.nan)
    return up_markers, down_markers

#Generating Colors For Histogram
def gen_macd_color(df):
    macd_color = []
    macd_color.clear()
    for i in range (0,len(df['PPO_HIST_8'][-252:])):
        if df['PPO_HIST_8'][-252:][i] >= 0 and df['PPO_HIST_8'][-252:][i-1] < df['PPO_HIST_8'][-252:][i]:
            macd_color.append('#2e9e18')
            #print(i,'green')
        elif df['PPO_HIST_8'][-252:][i] >= 0 and df['PPO_HIST_8'][-252:][i-1] > df['PPO_HIST_8'][-252:][i]:
            macd_color.append('#5fd448')
            #print(i,'faint green')
        elif df['PPO_HIST_8'][-252:][i] < 0 and df['PPO_HIST_8'][-252:][i-1] >df['PPO_HIST_8'][-252:][i] :
            #print(i,'red')
            macd_color.append('#ed1e13')
        elif df['PPO_HIST_8'][-252:][i] < 0 and df['PPO_HIST_8'][-252:][i-1] < df['PPO_HIST_8'][-252:][i] :
            #print(i,'faint red')
            macd_color.append('#ff7770')
        else:
            macd_color.append('#000000')
            #print(i,'no')
    return macd_color



up_markers, down_markers = detect_intraday_price_diff(df[['Close','Low','High','LABEL1']][-252:])
up_plot = fplt.make_addplot(up_markers, type='scatter',panel=3, marker=r'$\Uparrow$', markersize=200, color='green',secondary_y='auto',label="Predicted BUY")
down_plot = fplt.make_addplot(down_markers, type='scatter',panel=3, marker=r'$\downarrow$', markersize=200, color='red',secondary_y='auto',label="Predicted SELL")

panel2= fplt.make_addplot((df['BB_P_12'][-252:]),panel=0,color='green',alpha=1,linestyle='solid',ylabel='BB',label='BB_P_12')
panel22= fplt.make_addplot((df['BB_P_20'][-252:]),panel=0,color='red',alpha=1,linestyle='solid',ylabel='',label='BB_P_20')
fill_between=[BB_fill_up,BB_fill_down]
#panel1= fplt.make_addplot((df['BB_P_6'][-252:]),panel=0,color='black',alpha=1,linestyle='solid',ylabel='BB_P',fill_between=[BB_fill_up,BB_fill_down])
#panel2= fplt.make_addplot((df['AROON_DOWN_7'][-252:]),panel=0,color='black',alpha=1,linestyle='--',ylabel='AROON',label="AROON_DOWN_7",fill_between=[ARU_fill_up,ARU_fill_down])
#panel29= fplt.make_addplot((df['AROON_9'][-252:]),panel=8,color='black',alpha=1,linestyle='--',ylabel='AROON',label="AROON_9",fill_between=[ARU_fill_up,ARU_fill_down])
panel24= fplt.make_addplot(list(map(lambda i: 2 if i==1 else -1 ,df['LABEL'][-252:])),panel=0, color='black',linestyle='-')
panel15= fplt.make_addplot((df['LABEL'][-252:]),panel=3, color='black',linestyle='-',label="Labeling Fluctuation")
#=======================================================================================================================================
panel6= fplt.make_addplot((df['KC_P_9'][-252:]),panel=2,color='green',alpha=1,linestyle='solid',ylabel='',label="KC_P_9")
panel4= fplt.make_addplot((df['KC_P_16'][-252:]),panel=2,color='red',alpha=1,linestyle='solid',ylabel='KC',label="KC_P_16")
fill_between=[KC_fill_up,KC_fill_down]
#panel18= fplt.make_addplot((df['DC_P_18'][-252:]),panel=1,color='red',alpha=1,linestyle='solid',ylabel='DC',label="DC_P_18")
panel16= fplt.make_addplot(list(map(lambda i: -2 if i==0 else 3 ,df['LABEL'][-252:])),panel=1, color='black',linestyle='solid')
#=======================================================================================================================================
#panel5= fplt.make_addplot((df['SLOWD_6'][-252:]/100),panel=1,color='black',alpha=1,linestyle='--',ylabel="SO",label="SLOWD_6",fill_between=[SLOW_fill_up,SLOW_fill_down])
#panel4= fplt.make_addplot((df['SLOWK_13'][-252:]/100),panel=1,color='red',alpha=1,linestyle='solid',ylabel="",label="SLOWK_13")
fill_between=[SLOW_fill_up,SLOW_fill_down]
panel5= fplt.make_addplot((df['ULTOSC_9'][-252:]),panel=1,color='black',alpha=1,linestyle='--',ylabel="ULTOSC",label="ULTOSC_9",fill_between=[UO_fill_up,UO_fill_down])
#panel6= fplt.make_addplot((df['ULTOSC_15'][-252:]),panel=2,color='black',alpha=1,linestyle='--',ylabel="ULTOSC",label="ULTOSC_15",fill_between=[UO_fill_up,UO_fill_down])

panel19= fplt.make_addplot(list(map(lambda i: -2 if i==0 else 3 ,df['LABEL'][-252:])),panel=2, color='black',linestyle='solid')
#=======================================================================================================================================
panel7= fplt.make_addplot(df['DC_P_18'][-252:],panel=4,color='black',alpha=1,linestyle='--',ylabel='DC',label='DC_P_18',secondary_y='auto',fill_between=[DC_fill_up,DC_fill_down])
#panel18= fplt.make_addplot(df['DC_P_12'][-252:],panel=4,color='red',alpha=1,linestyle='solid',ylabel='',label='DC_P_12',secondary_y='auto')
fill_between=[DC_fill_up,DC_fill_down]
fill_between=[TSI_fill_up,TSI_fill_down]
#panel23= fplt.make_addplot((df['LABEL'][-252:]),panel=5, color='black',linestyle='solid')
#panel7= fplt.make_addplot(df['PPO_HIST_8'][-252:],panel=4,type='bar',color=gen_macd_color(df),alpha=1,linestyle='solid',ylabel='PPO',label='PPO_HIST_8',secondary_y='auto')
panel30= fplt.make_addplot(list(map(lambda i: 0 if i==0 else 50 ,df['LABEL'][-252:])),panel=7, color='black',linestyle='solid')
panel28= fplt.make_addplot(list(map(lambda i: -200 if i==0 else 200 ,df['LABEL'][-252:])),panel=6, color='black',linestyle='solid')
#=======================================================================================================================================
panel10= fplt.make_addplot(df['WILLIAM_7'][-252:],panel=5,color='black',alpha=1,linestyle='--',ylabel='WILLIAM_7',label='WILLIAM_7',fill_between=[WILLIAM_fill_up,WILLIAM_fill_down])
#panel23= fplt.make_addplot(df['WIllIAM_20'][-252:],panel=5,color='red',alpha=1,linestyle='solid',ylabel='WIllIAM',label='WIllIAM_20')
fill_between=[WILLIAM_fill_up,WILLIAM_fill_down]
#panel10= fplt.make_addplot((df['LABEL'][-252:]),panel=5, color='black',linestyle='solid')
panel27= fplt.make_addplot(df['CCI_7'][-252:],panel=6,color='green',alpha=1,linestyle='solid',ylabel='',label='CCI_7',secondary_y='auto')
panel3= fplt.make_addplot(df['CCI_10'][-252:],panel=6,color='red',alpha=1,linestyle='solid',ylabel='CCI',label='CCI_10',secondary_y='auto')

fill_between=[CCI_fill_up,CCI_fill_down]
#panel18= fplt.make_addplot(df['CCI_19'][-252:],panel=6,color='red',alpha=1,linestyle='solid',ylabel='',label='CCI_19',secondary_y='auto')
fill_between=[WILLIAM_fill_up,WILLIAM_fill_down]
#======================================================================================================================================
#panel8= fplt.make_addplot(df['RSI_7'][-252:],panel=5,color='black',alpha=1,linestyle='--',ylabel='RSI',label='RSI_7',fill_between=[RSI_fill_up,RSI_fill_down])
#panel7= fplt.make_addplot(df['RSI_14'][-252:],panel=4,color='black',alpha=1,linestyle='--',ylabel='RSI',label='RSI_14',fill_between=[RSI_fill_up,RSI_fill_down])
fill_between=[RSI_fill_up,RSI_fill_down]
panel11= fplt.make_addplot(list(map(lambda i: 70 if i==1 else 0 ,df['LABEL'][-252:])),panel=5, color='black',linestyle='solid')
panel12= fplt.make_addplot(df['ADX_POS_12'][-252:],panel=7,color='green',alpha=1,linestyle='solid',ylabel='ADX',label='ADX_POS_12')
panel29= fplt.make_addplot(df['ADX_NEG_8'][-252:],panel=7,color='red',alpha=1,linestyle='solid',ylabel='',label='ADX_NEG_8')
fill_between=[ADXP_fill_up,ADXP_fill_down]
fill_between=[ADXP_fill_up,ADXP_fill_down]
panel25= fplt.make_addplot(df['LABEL'][-252:],panel=4, color='black',linestyle='solid')
#=======================================================================================================================================
#panel7= fplt.make_addplot(df['CMO_6'][-252:],panel=4,color='green',alpha=1,linestyle='solid',ylabel='CMO',label='CMO_6')
#panel18= fplt.make_addplot(df['CMO_14'][-252:],panel=4,color='red',alpha=1,linestyle='solid',ylabel='',label='CMO_14')
panel21= fplt.make_addplot(df['LABEL'][-252:],panel=5, color='black',linestyle='solid')
fill_between=[CMO_fill_up,CMO_fill_down]
#panel10= fplt.make_addplot(df['MACD_DIFF_9'][-252:],panel=6,color='black',alpha=1,linestyle='--',ylabel='MACD',label='MACD_DIFF_9',fill_between=[MACD_fill_up,MACD_fill_down])

#panel29= fplt.make_addplot(df['TSI_6'][-252:],panel=7,color='black',alpha=1,linestyle='--',ylabel='TSI',label='TSI_6',fill_between=[TSI_fill_up,TSI_fill_down])

#panel25= fplt.make_addplot(list(map(lambda i: -100 if i==0 else 0 ,df['LABEL'][-252:])),panel=5, color='black',linestyle='solid')
#panel20= fplt.make_addplot(df['ADX_POS_13'][-252:],panel=6,color='red',alpha=1,linestyle='solid',ylabel='ADX',label='ADX_POS_13',secondary_y='auto')

#=======================================================================================================================================

mc = fplt.make_marketcolors(
                            up='tab:green',down='tab:red',
                            edge='b',
                            wick={'up':'green','down':'red'},
                            volume='lawngreen',
                           )
s  = fplt.make_mpf_style(marketcolors=mc)

fig, axlist=fplt.plot(
          df[-252:],
          type='candle', volume=False,update_width_config=dict(candle_linewidth=1),#hlines=dict(hlines=[130,150]),
          addplot = [up_plot,down_plot,panel2,panel3,panel4,panel5,panel6,panel7,panel10,panel11,panel15,panel12,panel16,panel19,panel22,panel24,panel25,panel27,panel28,panel29,panel30],
          figratio=(40,30),
          panel_ratios=(1,1,1,5,1,1,1,1),
          figscale=2,
          tight_layout=False,
          style='charles',
          #title='Salesforce, Inc. \n\n\n\n',
          ylabel='Price ($)',
          #mav=(3,6,9),
          main_panel=3,
          num_panels=8,
          returnfig=True,
          #savefig="/content/sample_data/candle_.svg"
)
# add a new suptitle
#fig.suptitle('Franklin Resources, Inc. (BEN)', y=0.92, fontsize=25, x=0.55)

# add a title the the correct axes
axlist[0].set_title('', fontsize=25, style='italic', fontfamily='Helvetica', loc='center')
plt.show()

# save the figure
fig.savefig('/content/sample_data/candle_DVN.svg', bbox_inches='tight')
