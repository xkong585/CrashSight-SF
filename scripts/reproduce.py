"""Recompute crash summaries, an out-of-time severity baseline, and siting scenarios.

This portable workflow complements the original Colab notebooks. It deliberately
does not present the original random grid-split AUC as future-risk performance.
"""
from pathlib import Path
import argparse, hashlib, json, platform, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import StrMethodFormatter, MaxNLocator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve
from scipy.spatial.distance import cdist
import folium

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'refreshed'
NAVY, TEAL, GOLD, RED = '#17324D', '#168B8B', '#D59B36', '#C46459'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
    'axes.spines.top':False, 'axes.spines.right':False, 'axes.labelcolor':NAVY,
    'text.color':NAVY, 'axes.titleweight':'bold', 'figure.facecolor':'#F6F8FA',
    'axes.facecolor':'white', 'savefig.facecolor':'#F6F8FA', 'axes.edgecolor':'#CBD5DF'})
CMAP = LinearSegmentedColormap.from_list('ocean', ['#F3F8F7', '#8DD0C5', TEAL, NAVY])

def save(fig, name, title, subtitle):
    fig.suptitle(title, x=.06, ha='left', fontsize=18, fontweight='bold', y=.99)
    fig.text(.06, .93, subtitle, fontsize=10, color='#53687A')
    fig.tight_layout(rect=[.02,.04,.98,.88], h_pad=2.5, w_pad=2.5)
    fig.text(.06,.015,'Source: SF injury-crash extract dated 2025-04-21 | Recomputed from source records',fontsize=8,color='#64748B')
    fig.savefig(OUT/name, dpi=170)
    plt.close(fig)

def chart_counts(ax, values, color=TEAL, horizontal=False):
    if horizontal:
        values=values.sort_values(); ax.barh(values.index.astype(str), values.values, color=color)
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'));ax.set_xlabel('Recorded crashes')
        ax.xaxis.set_major_locator(MaxNLocator(4))
    else:
        ax.bar(values.index.astype(str), values.values,color=color)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'));ax.set_ylabel('Recorded crashes')
    ax.grid(axis='x' if horizontal else 'y', alpha=.16);ax.set_axisbelow(True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=ROOT/'data/Traffic_Crashes_Resulting_in_Injury_20250421.zip')
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.data) as z:
        name=next(n for n in z.namelist() if n.endswith('.csv'))
        raw=pd.read_csv(z.open(name),low_memory=False)
    data=raw.drop_duplicates('case_id_pkey').copy()
    data['timestamp']=pd.to_datetime(data.collision_datetime,format='%m/%d/%Y %I:%M:%S %p',errors='coerce')
    valid_dates=data.timestamp.notna(); data=data.loc[valid_dates].copy()
    data['year']=data.timestamp.dt.year;data['month']=data.timestamp.dt.month
    data['hour']=data.timestamp.dt.hour;data['weekday']=data.timestamp.dt.dayofweek
    summaries={'year':data.groupby('year').size(),'month':data.groupby('month').size(),
               'hour':data.groupby('hour').size(),'weekday':data.groupby('weekday').size()}
    for key,s in summaries.items():s.rename('count').to_csv(OUT/f'{key}_counts.csv')
    years=summaries['year'];fig,axs=plt.subplots(2,2,figsize=(12,8))
    axs[0,0].plot(years.index,years.values,color=TEAL,lw=2.5,marker='o',ms=4)
    axs[0,0].set_title('Annual records');axs[0,0].set_ylabel('Recorded crashes');axs[0,0].set_xticks([2005,2010,2015,2020,2025]);axs[0,0].grid(alpha=.16)
    axs[0,0].annotate('2025 is a partial year',xy=(years.index[-1],years.iloc[-1]),xytext=(-130,35),textcoords='offset points',arrowprops={'arrowstyle':'->','color':GOLD},color='#936419')
    chart_counts(axs[0,1],summaries['month'].reindex(range(1,13),fill_value=0));axs[0,1].set_title('Month across all years');axs[0,1].set_xlabel('Month')
    wk=summaries['weekday'].reindex(range(7),fill_value=0);wk.index=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    chart_counts(axs[1,0],wk);axs[1,0].set_title('Day of week')
    chart_counts(axs[1,1],summaries['hour'].reindex(range(24),fill_value=0),GOLD);axs[1,1].set_title('Hour of day');axs[1,1].set_xticks(range(0,24,3))
    save(fig,'temporal_summary.png','When are injury crashes recorded?',f'{len(data):,} unique dated records | {data.timestamp.min():%Y-%m-%d} to {data.timestamp.max():%Y-%m-%d}')
    pivot=pd.crosstab(data.weekday,data.hour).reindex(index=range(7),columns=range(24),fill_value=0)
    pivot.to_csv(OUT/'weekday_hour_counts.csv')
    fig,ax=plt.subplots(figsize=(12,5));im=ax.imshow(pivot,cmap=CMAP,aspect='auto')
    ax.set_yticks(range(7),['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']);ax.set_xticks(range(0,24,2));ax.set_xlabel('Hour of day');fig.colorbar(im,ax=ax,label='Recorded crashes')
    save(fig,'weekday_hour.png','The weekly rhythm of recorded crashes','Counts describe the extract; they are not adjusted for traffic volume.')
    fig,axs=plt.subplots(2,2,figsize=(13,9))
    for ax,col,title in zip(axs.flat,['weather_1','lighting','road_surface','type_of_collision'],['Weather','Lighting','Road surface','Collision type']):
        chart_counts(ax,data[col].fillna('Not recorded').value_counts().head(6),horizontal=True);ax.set_title(title)
    save(fig,'collision_context.png','Conditions and collision characteristics','Frequent conditions are not necessarily higher-risk conditions: exposure denominators are unavailable.')
    fig,ax=plt.subplots(figsize=(11,7));counts=data.analysis_neighborhood.fillna('Not recorded').value_counts().head(15)
    counts.rename('count').to_csv(OUT/'top_neighborhoods.csv');chart_counts(ax,counts,horizontal=True)
    save(fig,'neighborhoods.png','Where are the most crashes recorded?','Top 15 reported neighborhoods | Counts reflect both exposure and reporting patterns.')
    geo=data[data.tb_latitude.between(37.7,37.85)&data.tb_longitude.between(-122.52,-122.36)].copy()
    fig,ax=plt.subplots(figsize=(9,8));h=ax.hexbin(geo.tb_longitude,geo.tb_latitude,gridsize=60,cmap=CMAP,mincnt=1,bins='log')
    ax.set_xlabel('Longitude');ax.set_ylabel('Latitude');ax.set_aspect(1/np.cos(np.radians(37.77)));fig.colorbar(h,ax=ax,label='Crash count (log scale)')
    save(fig,'spatial_density.png','Crash concentration across San Francisco',f'{len(geo):,} records inside the stated SF coordinate bounds | No inferred road-level risk')

    # A distinct, explicitly documented baseline: severity conditional on a recorded
    # injury crash. Features exclude casualties, severity labels, and target-derived density.
    data['severe']=data.collision_severity.isin(['Fatal','Injury (Severe)']).astype(int)
    cats=['weather_1','lighting','road_surface','type_of_collision','analysis_neighborhood']
    nums=['hour','weekday','month']
    model=Pipeline([('features',ColumnTransformer([
        ('category',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='Not recorded')),('encode',OneHotEncoder(handle_unknown='ignore'))]),cats),
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums)])),
        ('classifier',LogisticRegression(max_iter=1500,random_state=42))])
    train=data[data.year<=2022];validation=data[data.year==2023];test=data[data.year==2024]
    assert train.timestamp.max()<validation.timestamp.min()<test.timestamp.min()
    model.fit(train[cats+nums],train.severe);probs=model.predict_proba(test[cats+nums])[:,1]
    auc=roc_auc_score(test.severe,probs);ap=average_precision_score(test.severe,probs)
    # Choose a threshold on 2023 only; 2024 remains untouched until evaluation.
    vp=model.predict_proba(validation[cats+nums])[:,1]
    precision,recall,thresholds=precision_recall_curve(validation.severe,vp)
    f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    threshold=float(thresholds[np.argmax(f1)])
    cm=confusion_matrix(test.severe,probs>=threshold,labels=[0,1]);fpr,tpr,_=roc_curve(test.severe,probs)
    pd.DataFrame({'case_id':test.case_id_pkey,'actual_severe':test.severe,'predicted_probability':probs}).to_csv(OUT/'heldout_predictions.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(11,5));axs[0].plot(fpr,tpr,color=TEAL,lw=2.5,label=f'Logistic baseline: AUC {auc:.3f}')
    axs[0].plot([0,1],[0,1],color='#94A3B8',ls='--');axs[0].legend(loc='lower right',fontsize=9);axs[0].set(xlabel='False positive rate',ylabel='True positive rate',title='Out-of-time ROC')
    axs[1].imshow(cm,cmap=CMAP)
    for i in range(2):
        for j in range(2):axs[1].text(j,i,f'{cm[i,j]:,}',ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else NAVY,fontsize=16)
    axs[1].set(xticks=[0,1],yticks=[0,1],xticklabels=['Other injury','Severe / fatal'],yticklabels=['Other injury','Severe / fatal'],xlabel='Predicted',ylabel='Recorded',title=f'2023 validation threshold: {threshold:.3f}')
    save(fig,'severity_baseline.png','Can recorded context distinguish crash severity?',f'Train: through 2022 | Threshold: 2023 F1 | Test: 2024, n={len(test):,} | AP {ap:.3f}; prevalence {test.severe.mean():.3f}')

    # Deterministic siting scenario anchored to the dataset, not the machine's date.
    end=geo.timestamp.max();recent=geo[geo.timestamp>=end-pd.DateOffset(months=12)].copy()
    lat0=37.77;lon0=-122.44;cos=np.cos(np.radians(lat0))
    recent['gx']=np.floor((recent.tb_longitude-lon0)*111.32*cos/.5).astype(int)
    recent['gy']=np.floor((recent.tb_latitude-lat0)*111.32/.5).astype(int)
    grid=recent.groupby(['gx','gy']).agg(weight=('case_id_pkey','size'),latitude=('tb_latitude','mean'),longitude=('tb_longitude','mean')).reset_index()
    candidates=grid.sort_values(['weight','gx','gy'],ascending=[False,True,True]).head(40).copy()
    coords=lambda t:np.column_stack(((t.longitude-lon0)*111.32*cos,(t.latitude-lat0)*111.32))
    distances=cdist(coords(grid),coords(candidates));weights=grid.weight.to_numpy()
    chosen=[];nearest=np.full(len(grid),np.inf);comparisons=[]
    for k in range(12):
        available=[j for j in range(len(candidates)) if j not in chosen]
        pick=min(available,key=lambda j:float(np.dot(np.minimum(nearest,distances[:,j]),weights)))
        chosen.append(pick);nearest=np.minimum(nearest,distances[:,pick])
        if k+1 in [8,10,12]:comparisons.append({'facilities':k+1,'weighted_mean_distance_km':float(np.average(nearest,weights=weights)),'within_1km':float(np.average(nearest<=1,weights=weights)),'within_2km':float(np.average(nearest<=2,weights=weights))})
    sites=candidates.iloc[chosen[:10]].copy();sites.to_csv(OUT/'selected_sites.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'facility_comparison.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(12,6));axs[0].scatter(grid.longitude,grid.latitude,s=12+grid.weight*1.1,c=grid.weight,cmap=CMAP,alpha=.65)
    axs[0].scatter(sites.longitude,sites.latitude,marker='*',s=170,c=GOLD,edgecolors=NAVY,label='10 selected candidates');axs[0].legend(fontsize=8);axs[0].set(xlabel='Longitude',ylabel='Latitude',title='Illustrative candidate locations');axs[0].xaxis.set_major_locator(MaxNLocator(4))
    comp=pd.DataFrame(comparisons);axs[1].plot(comp.facilities,comp.weighted_mean_distance_km,marker='o',lw=2.5,color=TEAL);axs[1].set(xticks=[8,10,12],xlabel='Facilities selected',ylabel='Crash-weighted distance (km)',title='Greedy facility-location scenarios');axs[1].grid(alpha=.2)
    save(fig,'facility_scenarios.png','A transparent resource-location scenario',f'12 months ending {end:%Y-%m-%d} | 500 m grid | Top 40 occupied cells as candidates | Greedy heuristic, not a certified optimum')
    m=folium.Map(location=[37.77,-122.43],zoom_start=12,tiles='OpenStreetMap')
    for row in grid.itertuples():folium.CircleMarker([row.latitude,row.longitude],radius=3+min(row.weight/10,8),color=TEAL,fill=True,fill_opacity=.35,weight=1,tooltip=f'{row.weight} recorded crashes').add_to(m)
    for i,row in enumerate(sites.itertuples()):folium.Marker([row.latitude,row.longitude],tooltip=f'Illustrative site {i+1}',icon=folium.Icon(color='orange',icon='info-sign')).add_to(m)
    m.save(OUT/'facility_map.html')
    summary={'input_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),'raw_records':len(raw),'unique_dated_records':len(data),'coordinate_valid_records':len(geo),'date_min':str(data.timestamp.min()),'date_max':str(data.timestamp.max()),'duplicates_removed':len(raw)-len(raw.drop_duplicates('case_id_pkey')),'invalid_dates':int((~valid_dates).sum()),'model':{'task':'Severe/fatal versus other injury among recorded injury crashes','train_records':len(train),'validation_records':len(validation),'test_records':len(test),'train_years':'through 2022','validation_year':2023,'test_year':2024,'roc_auc':auc,'average_precision':ap,'test_prevalence':float(test.severe.mean()),'threshold':threshold,'threshold_selection':'Maximum F1 on 2023 validation set only','confusion_matrix':cm.tolist()},'facility_scenarios':comparisons,'python':platform.python_version(),'scope':'New portable EDA, temporal severity baseline, and deterministic siting scenario. Original Colab grid-risk, Moran/LISA and TSP outputs not revalidated.'}
    (OUT/'run_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
