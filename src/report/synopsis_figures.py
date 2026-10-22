import pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
BLUE,GREY,OCHRE='#2f5d8a','#8a96a3','#b07a2a'
y=pd.read_csv('../../data/interim/phonepe_state_year.csv')
st=pd.read_csv('../../config/states.csv')
y=y.merge(st[['state_code','region']],on='state_code')
# Figure 2: national + mix
nat=y.groupby('year')[['upi_txn','upi_txn_p2p','upi_txn_retail','upi_txn_utility']].sum()
sh=nat[['upi_txn_p2p','upi_txn_retail','upi_txn_utility']].div(nat.upi_txn,axis=0)*100
fig,ax=plt.subplots(1,2,figsize=(6.6,2.7),dpi=220)
ax[0].bar(nat.index,nat.upi_txn/1e9,color=BLUE,width=.65)
for x,v in zip(nat.index,nat.upi_txn/1e9): ax[0].text(x,v+2,f'{v:.1f}',ha='center',fontsize=7)
ax[0].set_title('(a) Transactions per year (billions)',loc='left',fontsize=8.5); ax[0].set_xticks(nat.index); ax[0].tick_params(axis='x',labelsize=7,rotation=45)
for c,col,l in [('upi_txn_retail',BLUE,'Merchant/retail'),('upi_txn_p2p',GREY,'Person-to-person'),('upi_txn_utility',OCHRE,'Utility/bills')]:
    ax[1].plot(sh.index,sh[c],marker='o',ms=3,color=col,label=l)
ax[1].set_title('(b) Share of transactions (%)',loc='left',fontsize=8.5); ax[1].legend(frameon=False,fontsize=7); ax[1].set_xticks(sh.index); ax[1].tick_params(axis='x',labelsize=7,rotation=45)
fig.tight_layout(); fig.savefig(f'../../reports/figures/synopsis_fig2.png')
# Figure 3 concentration
top5=y.groupby('year').apply(lambda d:d.nlargest(5,'upi_txn').upi_txn.sum()/d.upi_txn.sum()*100,include_groups=False)
ne=y[y.region=='North-East'].groupby('year').upi_txn.sum()/y.groupby('year').upi_txn.sum()*100
fig,ax=plt.subplots(1,2,figsize=(6.6,2.5),dpi=220)
ax[0].plot(top5.index,top5,marker='o',ms=3,color=BLUE); ax[0].set_title('(a) Five largest states (% of total)',loc='left',fontsize=8.5)
ax[1].plot(ne.index,ne,marker='o',ms=3,color=GREY); ax[1].set_title('(b) Eight North-Eastern states (% of total)',loc='left',fontsize=8.5)
for a in ax: a.set_xticks(top5.index); a.tick_params(axis='x',labelsize=7,rotation=45)
fig.tight_layout(); fig.savefig(f'../../reports/figures/synopsis_fig3.png')
print(top5.round(1).to_dict(), ne.round(2).to_dict(), (nat.upi_txn/1e9).round(1).to_dict(), sh.round(1).iloc[[0,-1]].to_dict())
# Figure 1 framework
fig,ax=plt.subplots(figsize=(6.6,3.9),dpi=220); ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(0,7)
def box(x,y,w,h,title,lines,fc,ls='-'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.1',fc=fc,ec='#333',lw=.8,ls=ls))
    ax.text(x+w/2,y+h-.22,title,ha='center',va='top',fontsize=8.3,weight='bold')
    ax.text(x+w/2,y+h-.62,'\n'.join(lines),ha='center',va='top',fontsize=7.3,linespacing=1.3)
box(.05,1.6,2.75,3.6,'Digital adoption',['ln UPI transactions per capita','ln e-Taal transactions per capita','ln DBT per capita','Internet subscribers per 100','→ adoption index (z)'],'#dce6f1')
box(3.55,5.45,2.7,1.45,'Moderators',['Literacy (Census 2011)','ln NSDP per capita'],'#f2f2f2')
box(7.0,4.55,2.95,2.35,'Service delivery',['Primary: % MGNREGS wages','paid within 15 days','Secondary: SDEI (wages +','ePoS authentication)'],'#e2efda')
box(7.0,2.2,2.95,2.05,'Inclusion',["Primary: women's share of","MGNREGS person-days","Secondary: PMJDY women /","rural share"],'#e2efda')
box(7.0,0.05,2.95,1.85,'Online perception',['Net sentiment of public posts','(secondary, gated)'],'#ffffff',ls='--')
kw=dict(arrowstyle='-|>',color='#333',lw=1.1,mutation_scale=11)
ax.annotate('',xy=(6.95,4.85),xytext=(2.85,4.85),arrowprops=kw); ax.text(4.9,4.72,'RQ1',fontsize=7.5,style='italic',ha='center',va='top')
ax.annotate('',xy=(4.9,4.9),xytext=(4.9,5.45),arrowprops=kw); ax.text(5.0,5.15,'RQ2',fontsize=7.5,style='italic',va='center')
ax.annotate('',xy=(6.95,3.2),xytext=(2.85,3.2),arrowprops=kw); ax.text(4.9,3.28,'RQ4',fontsize=7.5,style='italic',ha='center')
ax.annotate('',xy=(6.95,1.0),xytext=(2.85,2.0),arrowprops=dict(arrowstyle='<|-|>',color='#333',lw=1,ls='--',mutation_scale=11)); ax.text(4.9,1.2,'RQ3',fontsize=7.5,style='italic',ha='center')
ax.text(5.0,0.2,'All models: state and year fixed effects,\nln NSDP control, state-clustered SEs',ha='center',fontsize=7,color='#444')
fig.tight_layout(); fig.savefig(f'../../reports/figures/synopsis_fig1.png')
