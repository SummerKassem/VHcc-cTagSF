outDir = "output_2017_PFNano" #"190928_2017"

## 2018
# DYPath = "/nfs/dust/cms/user/spmondal/ctag_condor/200113_all_2018_DY/"
# TTPath = "/nfs/dust/cms/user/spmondal/ctag_condor/200114_all_2018_TT/"
# WcPath = "/nfs/dust/cms/user/spmondal/ctag_condor/200109_all_2018_Wc/"

# 2017
DYPath = "/nfs/dust/cms/user/summer/ctag_condor/2022_June_2017_DY_200/"
WcPath = "/nfs/dust/cms/user/summer/ctag_condor/2022_June_2017_Wc_200/"
#TTPath = "/nfs/dust/cms/user/spmondal/ctag_condor/190924_final_2017_TT/"
#WcPath = "/nfs/dust/cms/user/spmondal/ctag_condor/190924_final_2017_Wc/"
#WcPath = "/nfs/dust/cms/user/anstein/ctag_condor/210402_2017_Wc_minimal/"

# 2016
#DYPath = "/nfs/dust/cms/user/spmondal/ctag_condor/190621_2016_DY/"
#TTPath = "/nfs/dust/cms/user/spmondal/ctag_condor/190222_TT_pt20_Alltrigs/"
#WcPath = "/nfs/dust/cms/user/spmondal/ctag_condor/190228_pt20/"

#2016Nanov7
# DYPath = "/nfs/dust/cms/user/spmondal/ctag_condor/201030_2016Nanov7_DY/"
# TTPath = "/nfs/dust/cms/user/spmondal/ctag_condor/201030_2016Nanov7_TT/"
# WcPath = "/nfs/dust/cms/user/spmondal/ctag_condor/201030_2016Nanov7_Wc/"


# 2017 PU Loose
# DYPath = "/nfs/dust/cms/user/spmondal/ctag_condor/200706_all_2017_PULoose_DY/"
# TTPath = "/nfs/dust/cms/user/spmondal/ctag_condor/200706_all_2017_PULoose_TT/"
# WcPath = "/nfs/dust/cms/user/spmondal/ctag_condor/200706_all_2017_PULoose_Wc/"

systs = [
         "central",
         "PUWeight_up","PUWeight_down","MuIDSF_up","MuIDSF_down","EleIDSF_up","EleIDSF_down",
         "LHEScaleWeight_muR_up","LHEScaleWeight_muR_down","LHEScaleWeight_muF_up","LHEScaleWeight_muF_down",
         "PSWeightISR_up","PSWeightISR_down","PSWeightFSR_up","PSWeightFSR_down",
         #"jesTotalUp","jesTotalDown","jerUp","jerDown",
         "XSec_WJets_up","XSec_WJets_down","XSec_DYJets_up","XSec_DYJets_down","XSec_ST_up","XSec_ST_down",
         "XSec_ttbar_up", "XSec_ttbar_down",
         "XSec_BRUnc_DYJets_b_up","XSec_BRUnc_DYJets_b_down","XSec_BRUnc_DYJets_c_up","XSec_BRUnc_DYJets_c_down","XSec_BRUnc_WJets_c_up","XSec_BRUnc_WJets_c_down"
         ]

SFfilesDeepCSV = ["","DeepCSV_ctagSF_MiniAOD94X_2017_pTincl_v3_2.root"]
SFfilesDeepJet = ["","DeepJet_ctagSF_MiniAOD94X_2017_pTincl_v3_2.root"]
SFhistSuff = [""] #"_ValuesSystOnlyUp","_ValuesSystOnlyDown"]   # "" for nominal

plotExtra = False
plotsysts = False
plotBinSlices = True  # needed if one wants to derive SFs later
validateSFs = False
addsel = '' #'&& jet_CvsL[max(0.,muJet_idx)] > 0.8 && jet_CvsB[max(0.,muJet_idx)] > 0.1'
 #'&& jet_Pt[max(0.,muJet_idx)] > 80 && jet_Pt[max(0.,muJet_idx)] < 10000'

muBiasTestIndex = '(muJet_idx==0?1:0)'
# muBiasTestIndex = 'getBJetIdx(jet_DeepFlavCvsB,muJet_idx)'
# muBiasTestIndex = 'getCJetIdx(jet_DeepFlavCvsB,jet_DeepFlavCvsL,muJet_idx)'

nBinDisc = 60  # was 30 before, now using 60 to be consistent with previous runs (that also plotted the bin slices and therefore used 60 bins)
if plotBinSlices: nBinDisc = 60 

plotBinnedKins = False
normMCtodata = True
plot2D = False

outDir = outDir.rstrip('/')

def applyCuts(ln,reg=""):
    ln = ln.replace('ZMASSCUT','[85,95,\"invert\"]')
    ln = ln.replace('CVXBINNING','varBin1=[-0.2,0.,0.2,0.4,0.6,0.8,1.],varBin2=[-0.2,0.,0.2,0.4,0.6,0.8,1.]')
    ln = ln.replace('JETIDX',muBiasTestIndex)
    if "central" in syst:
        ln = ln.replace('TTSEMIWEIGHT','MCWeightName="eventWeightUnsigned",DataWeightName="eventWeightUnsigned",yTitle="Events",outDir="OUTDIR_SYSTNAME",rootPath="WCPATH"')
        #ln = ln.replace('TTWEIGHT','MCWeightName="eventWeight",DataWeightName="eventWeight",yTitle="Events",outDir="OUTDIR_SYSTNAME",rootPath="TTPATH"')
        ln = ln.replace('WCWEIGHT','MCWeightName="eventWeight",DataWeightName="eventWeight",yTitle="OS-SS Events",outDir="OUTDIR_SYSTNAME",rootPath="WCPATH"')
        ln = ln.replace('DYWEIGHT','MCWeightName="eventWeight",DataWeightName="eventWeight",yTitle="Events",outDir="OUTDIR_SYSTNAME",rootPath="DYPATH"')
    #elif "MCStat" in syst:
        #ln = ln.replace('TTSEMIWEIGHT','MCWeightName="eventWeightUnsigned",DataWeightName="eventWeightUnsigned",yTitle="Events",outDir="OUTDIR_SYSTNAME",MCStat="SYSTNAME",rootPath="WCPATH"')
        #ln = ln.replace('TTWEIGHT','MCWeightName="eventWeight",DataWeightName="eventWeight",yTitle="Events",outDir="OUTDIR_SYSTNAME",MCStat="SYSTNAME",rootPath="TTPATH"')
        #ln = ln.replace('WCWEIGHT','MCWeightName="eventWeight",DataWeightName="eventWeight",yTitle="OS-SS Events",outDir="OUTDIR_SYSTNAME",MCStat="SYSTNAME",rootPath="WCPATH"')
        #ln = ln.replace('DYWEIGHT','MCWeightName="eventWeight",DataWeightName="eventWeight",yTitle="Events",outDir="OUTDIR_SYSTNAME",rootPath="DYPATH",MCStat="SYSTNAME"')
    #elif "dataStat" in syst:
        #ln = ln.replace('TTSEMIWEIGHT','MCWeightName="eventWeightUnsigned",DataWeightName="eventWeightUnsigned",yTitle="Events",outDir="OUTDIR_SYSTNAME",dataStat="SYSTNAME",rootPath="WCPATH"')
        #ln = ln.replace('TTWEIGHT','MCWeightName="eventWeight",DataWeightName="eventWeight",yTitle="Events",outDir="OUTDIR_SYSTNAME",dataStat="SYSTNAME",rootPath="TTPATH"')
        #ln = ln.replace('WCWEIGHT','MCWeightName="eventWeight",DataWeightName="eventWeight",yTitle="OS-SS Events",outDir="OUTDIR_SYSTNAME",dataStat="SYSTNAME",rootPath="WCPATH"')
        #ln = ln.replace('DYWEIGHT','MCWeightName="eventWeight",DataWeightName="eventWeight",yTitle="Events",outDir="OUTDIR_SYSTNAME",rootPath="DYPATH",dataStat="SYSTNAME"')
    elif syst.startswith("je"):
        ln = ln.replace('TTSEMIWEIGHT','MCWeightName="eventWeightUnsigned",DataWeightName="eventWeightUnsigned",yTitle="Events",outDir="OUTDIR_SYSTNAME",pathSuff="_SYSTNAME",rootPath="WCPATH"')
        #ln = ln.replace('TTWEIGHT','MCWeightName="eventWeight",DataWeightName="eventWeight",yTitle="Events",outDir="OUTDIR_SYSTNAME",pathSuff="_SYSTNAME",rootPath="TTPATH"')
        ln = ln.replace('WCWEIGHT','MCWeightName="eventWeight",DataWeightName="eventWeight",yTitle="OS-SS Events",outDir="OUTDIR_SYSTNAME",pathSuff="_SYSTNAME",rootPath="WCPATH"')
        ln = ln.replace('DYWEIGHT','MCWeightName="eventWeight",DataWeightName="eventWeight",yTitle="Events",outDir="OUTDIR_SYSTNAME",rootPath="DYPATH",pathSuff="_SYSTNAME"')
    elif "XSec" in syst:
        ln = ln.replace('TTSEMIWEIGHT','MCWeightName="eventWeightUnsigned",DataWeightName="eventWeightUnsigned",yTitle="Events",outDir="OUTDIR_SYSTNAME",useXSecUnc="SYSTNAME",rootPath="WCPATH"')
        #ln = ln.replace('TTWEIGHT','MCWeightName="eventWeight",DataWeightName="eventWeight",yTitle="Events",outDir="OUTDIR_SYSTNAME",useXSecUnc="SYSTNAME",rootPath="TTPATH"')
        ln = ln.replace('WCWEIGHT','MCWeightName="eventWeight",DataWeightName="eventWeight",yTitle="OS-SS Events",outDir="OUTDIR_SYSTNAME",useXSecUnc="SYSTNAME",rootPath="WCPATH"')
        ln = ln.replace('DYWEIGHT','MCWeightName="eventWeight",DataWeightName="eventWeight",yTitle="Events",outDir="OUTDIR_SYSTNAME",rootPath="DYPATH",useXSecUnc="SYSTNAME"')
    # elif "LepID" in syst:
    #     direc = syst.split('_')[-1]
    #     # ln = ln.replace('TTWEIGHT','MCWeightName="eventWeightUnsigned*EleIDSF_DIRECTION",DataWeightName="eventWeightUnsigned",yTitle="OS+SS Events",outDir="OUTDIR_SYSTNAME"')
    #     # ln = ln.replace('TTWEIGHT','MCWeightName="eventWeightUnsigned*MuIDSF_DIRECTION",DataWeightName="eventWeightUnsigned",yTitle="OS+SS Events",outDir="OUTDIR_SYSTNAME"')
    #     ln = ln.replace('TTWEIGHT','MCWeightName="eventWeight*MuIDSF_DIRECTION*EleIDSF_DIRECTION",DataWeightName="eventWeight",yTitle="Events",outDir="OUTDIR_SYSTNAME"')
    #     ln = ln.replace('WCWEIGHT','MCWeightName="eventWeight*EleIDSF_DIRECTION",DataWeightName="eventWeight",yTitle="OS-SS Events",outDir="OUTDIR_SYSTNAME"')
    #     ln = ln.replace('WCWEIGHT','MCWeightName="eventWeight*MuIDSF_DIRECTION",DataWeightName="eventWeight",yTitle="OS-SS Events",outDir="OUTDIR_SYSTNAME"')
    #     ln = ln.replace('DYWEIGHT','MCWeightName="eventWeight*MuIDSF_DIRECTION",DataWeightName="eventWeight",yTitle="Events",outDir="OUTDIR_SYSTNAME",rootPath="DYPATH"')
    #     ln = ln.replace('DIRECTION',direc)
    else:
        ln = ln.replace('TTSEMIWEIGHT','MCWeightName="eventWeightUnsigned*SYSTNAME",DataWeightName="eventWeightUnsigned",yTitle="Events",outDir="OUTDIR_SYSTNAME",rootPath="WCPATH"')
        #ln = ln.replace('TTWEIGHT','MCWeightName="eventWeight*SYSTNAME",DataWeightName="eventWeight",yTitle="Events",outDir="OUTDIR_SYSTNAME",rootPath="TTPATH"')
        ln = ln.replace('WCWEIGHT','MCWeightName="eventWeight*SYSTNAME",DataWeightName="eventWeight",yTitle="OS-SS Events",outDir="OUTDIR_SYSTNAME",rootPath="WCPATH"')
        ln = ln.replace('DYWEIGHT','MCWeightName="eventWeight*SYSTNAME",DataWeightName="eventWeight",yTitle="Events",outDir="OUTDIR_SYSTNAME",rootPath="DYPATH"')
    
    ln = ln.replace('NBINDISC',str(nBinDisc))
    ln = ln.replace('OUTDIR',outDir)
    ln = ln.replace('SYSTNAME',syst)
    ln = ln.replace('DYPATH',DYPath)
    #ln = ln.replace('TTPATH',TTPath)
    ln = ln.replace('WCPATH',WcPath)

    ln = ln.replace('ESEL','filePre="Wc_e", selections="is_E == 1 && jetMuPt_by_jetPt < 0.6 && jet_nJet < 4 && diLepVeto == 0 QCDSELE UNISEL"')
    ln = ln.replace('MSEL','filePre="Wc_m", selections="is_M == 1 && jetMuPt_by_jetPt < 0.4 && jet_nJet < 4 && diLepVeto == 0 && (Z_Mass_best < 80 || Z_Mass_best > 100) && (Z_Mass_min > 12 || Z_Mass_min < 0) && jet_muplusneEmEF[muJet_idx] < 0.7 && jetMu_iso > 0.5 QCDSELM UNISEL"')
    #ln = ln.replace('MSEL','filePre="Wc_m", selections="is_M == 1 && jetMuPt_by_jetPt < 0.4 && jet_nJet < 4 && diLepVeto == 0 && Z_Mass_best > 0 &&  Z_Mass_min > 12 && jet_muplusneEmEF[muJet_idx] < 0.7 && jetMu_iso > 0.5 QCDSELM UNISEL"')
   
    ln = ln.replace('TTSEMISELE','filePre="TT_semie", selections="is_E == 1 && jetMuPt_by_jetPt < 0.6 && jet_nJet > 3 && diLepVeto == 0 && jetMu_Pt > 5 && jetMu_Pt < 25 QCDSELE UNISEL"')
    ln = ln.replace('TTSEMISELM','filePre="TT_semim", selections="is_M == 1 && jetMuPt_by_jetPt < 0.4 && jet_nJet > 3 && diLepVeto == 0 && (Z_Mass < 85 || Z_Mass > 95) && (Z_Mass > 12 || Z_Mass < 0) && jetMu_Pt > 5 && jetMu_Pt < 25 QCDSELM UNISEL"')

    ln = ln.replace('TTSELME','filePre="TT_me", selections="is_ME == 1 UNISEL"')
    ln = ln.replace('TTSELMM','filePre="TT_mm", selections="is_MM == 1 && (Z_Mass < 75 || Z_Mass > 105) && (Z_Mass > 12 || Z_Mass < 0) && met_Pt > 40 UNISEL"')
    ln = ln.replace('TTSELEE','filePre="TT_ee", selections="is_EE == 1 && (Z_Mass < 75 || Z_Mass > 105) && (Z_Mass > 12 || Z_Mass < 0) && met_Pt > 40 UNISEL"')
    
    ln = ln.replace('DYSELM','filePre="DY_m", selections="is_M == 1 && M_Pt[0] > 20 && Z_Pt > 15 UNISEL"')
    ln = ln.replace('DYSELE','filePre="DY_e", selections="is_E == 1 && E_Pt[0] > 27 && Z_Pt > 15 UNISEL"')
    
    ln = ln.replace('QCDSELE','&& E_RelIso[0] < 0.05 && (hardE_Jet_PtRatio  > 0.75 || hardE_Jet_PtRatio  < 0.) && abs(E_dz[0]) < 0.02 && abs(E_dxy[0]) < 0.01  && E_sip3d[0] < 2.5')
    ln = ln.replace('QCDSELM','&& M_RelIso[0] < 0.05 && (hardMu_Jet_PtRatio > 0.75 || hardMu_Jet_PtRatio < 0.) && abs(M_dz[0]) < 0.01 && abs(M_dxy[0]) < 0.002 && M_sip3d[0] < 2')
    
    ln = ln.replace('UNISEL',addsel)
    ln = ln.replace('UNICUT','')

    if not reg=="": ln = ln.replace('REG',reg)
    return ln

arguments = '''
           "jet_CvsL[muJet_idx]","CvsL",6,-0.1,1,MSEL,dataset="smu",brName2D=["jet_CvsB[muJet_idx]"],brLabel2="CvsB",nbins2=6,CVXBINNING,drawStyle="",makeROOT=True,WCWEIGHT
          # "jet_CvsL[muJet_idx]","CvsL",6,-0.1,1,ESEL,dataset="sele",brName2D=["jet_CvsB[muJet_idx]"],brLabel2="CvsB",nbins2=6,CVXBINNING,drawStyle="",makeROOT=True,WCWEIGHT

            "jet_CvsL[muJet_idx]","CvsL",6,-0.1,1,TTSEMISELM,dataset="smu",brName2D=["jet_CvsB[muJet_idx]"],brLabel2="CvsB",nbins2=6,CVXBINNING,drawStyle="",makeROOT=True,TTSEMIWEIGHT
          #  "jet_CvsL[muJet_idx]","CvsL",6,-0.1,1,TTSEMISELE,dataset="sele",brName2D=["jet_CvsB[muJet_idx]"],brLabel2="CvsB",nbins2=6,CVXBINNING,drawStyle="",makeROOT=True,TTSEMIWEIGHT
           
          #  "jet_CvsL[muJet_idx]","CvsL",6,-0.1,1,TTSELMM,dataset="dmu",brName2D=["jet_CvsB[muJet_idx]"],brLabel2="CvsB",nbins2=6,CVXBINNING,drawStyle="",makeROOT=True,TTWEIGHT
          #  "jet_CvsL[muJet_idx]","CvsL",6,-0.1,1,TTSELEE,dataset="deg",brName2D=["jet_CvsB[muJet_idx]"],brLabel2="CvsB",nbins2=6,CVXBINNING,drawStyle="",makeROOT=True,TTWEIGHT
          #  "jet_CvsL[muJet_idx]","CvsL",6,-0.1,1,TTSELME,dataset="mue",brName2D=["jet_CvsB[muJet_idx]"],brLabel2="CvsB",nbins2=6,CVXBINNING,drawStyle="",makeROOT=True,TTWEIGHT
           
            "jet_CvsL[0]","CvsL",6,-0.1,1,DYSELM,dataset="dmu",brName2D=["jet_CvsB[0]"],brLabel2="CvsB",nbins2=6,CVXBINNING,drawStyle="",makeROOT=True,DYWEIGHT
          #  "jet_CvsL[0]","CvsL",6,-0.1,1,DYSELE,dataset="deg",brName2D=["jet_CvsB[0]"],brLabel2="CvsB",nbins2=6,CVXBINNING,drawStyle="",makeROOT=True,DYWEIGHT
'''

plot1D = '''
        #    "jet_CvsL[muJet_idx]",r"Jet DeepCSV CvsL (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        #    "jet_CvsL[muJet_idx]",r"Jet DeepCSV CvsL (e)",NBINDISC,-0.2,1,ESEL,dataset="sele",makeROOT=True,WCWEIGHT
        #    "jet_CvsB[muJet_idx]",r"Jet DeepCSV CvsB (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        #    "jet_CvsB[muJet_idx]",r"Jet DeepCSV CvsB (e)",NBINDISC,-0.2,1,ESEL,dataset="sele",makeROOT=True,WCWEIGHT
           
        #    "jet_CvsL[muJet_idx]",r"Jet DeepCSV CvsL (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #    "jet_CvsL[muJet_idx]",r"Jet DeepCSV CvsL (e)",NBINDISC,-0.2,1,TTSEMISELE,dataset="sele",makeROOT=True,TTSEMIWEIGHT
        #    "jet_CvsB[muJet_idx]",r"Jet DeepCSV CvsB (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #    "jet_CvsB[muJet_idx]",r"Jet DeepCSV CvsB (e)",NBINDISC,-0.2,1,TTSEMISELE,dataset="sele",makeROOT=True,TTSEMIWEIGHT

        #    "W_Mass",r"m_{T}^{W} (mu)", 30,50,200,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
            
        #     "jet_CvsL[muJet_idx]",r"Jet DeepCSV CvsL (#mu #mu)",NBINDISC,-0.2,1,TTSELMM,dataset="dmu",makeROOT=True,TTWEIGHT
        #     "jet_CvsL[muJet_idx]",r"Jet DeepCSV CvsL (e e)",NBINDISC,-0.2,1,TTSELEE,dataset="deg",makeROOT=True,TTWEIGHT
        #     "jet_CvsL[muJet_idx]",r"Jet DeepCSV CvsL (#mu e)",NBINDISC,-0.2,1,TTSELME,dataset="mue",makeROOT=True,TTWEIGHT
        #     "jet_CvsB[muJet_idx]",r"Jet DeepCSV CvsB (#mu #mu)",NBINDISC,-0.2,1,TTSELMM,dataset="dmu",makeROOT=True,TTWEIGHT
        #     "jet_CvsB[muJet_idx]",r"Jet DeepCSV CvsB (e e)",NBINDISC,-0.2,1,TTSELEE,dataset="deg",makeROOT=True,TTWEIGHT
        #     "jet_CvsB[muJet_idx]",r"Jet DeepCSV CvsB (#mu e)",NBINDISC,-0.2,1,TTSELME,dataset="mue",makeROOT=True,TTWEIGHT
           
        #     "jet_CvsL[0]",r"Jet DeepCSV CvsL (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
        #     "jet_CvsB[0]",r"Jet DeepCSV CvsB (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
        #     "jet_CvsL[0]",r"Jet DeepCSV CvsL (e)",NBINDISC,-0.2,1,DYSELE,dataset="deg",makeROOT=True,DYWEIGHT
        #     "jet_CvsB[0]",r"Jet DeepCSV CvsB (e)",NBINDISC,-0.2,1,DYSELE,dataset="deg",makeROOT=True,DYWEIGHT

        #     "jet_btagDeepB[muJet_idx]",r"Jet DeepCSV P(b) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        #    "jet_btagDeepB[muJet_idx]",r"Jet DeepCSV P(b) (#mu e)",NBINDISC,-0.2,1,TTSELME,dataset="mue",makeROOT=True,TTWEIGHT
        #     "jet_btagDeepB[0]",r"Jet DeepCSV P(b) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
        
        
        
        #  "jet_CustomCvsL[muJet_idx]",r"Jet Custom CvsL (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        #  "jet_CustomCvsB[muJet_idx]",r"Jet Custom CvsB (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        #  "jet_CustomCvsL[muJet_idx]",r"Jet Custom CvsL (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #  "jet_CustomCvsB[muJet_idx]",r"Jet Custom CvsB (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #  "jet_CustomCvsL[0]",r"Jet Custom CvsL (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
        #  "jet_CustomCvsB[0]",r"Jet Custom CvsB (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
        
        #  "jet_CustomBvsL[muJet_idx]",r"Jet Custom BvsL (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        #  "jet_CustomBvsC[muJet_idx]",r"Jet Custom BvsC (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        #  "jet_CustomBvsL[muJet_idx]",r"Jet Custom BvsL (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #  "jet_CustomBvsC[muJet_idx]",r"Jet Custom BvsC (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #  "jet_CustomBvsL[0]",r"Jet Custom BvsL (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
        #  "jet_CustomBvsC[0]",r"Jet Custom BvsC (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      
        #  "jet_CustomProb_b[muJet_idx]",r"Jet Custom P(b) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        #  "jet_CustomProb_b[muJet_idx]",r"Jet Custom P(b) (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #  "jet_CustomProb_b[0]",r"Jet Custom P(b) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      
        #  "jet_CustomProb_bb[muJet_idx]",r"Jet Custom P(bb) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        #  "jet_CustomProb_bb[muJet_idx]",r"Jet Custom P(bb) (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #  "jet_CustomProb_bb[0]",r"Jet Custom P(bb) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      
        #  "jet_CustomProb_c[muJet_idx]",r"Jet Custom P(c) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        #  "jet_CustomProb_c[muJet_idx]",r"Jet Custom P(c) (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #  "jet_CustomProb_c[0]",r"Jet Custom P(c) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      
        #  "jet_CustomProb_l[muJet_idx]",r"Jet Custom P(udsg) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        #  "jet_CustomProb_l[muJet_idx]",r"Jet Custom P(udsg) (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #  "jet_CustomProb_l[0]",r"Jet Custom P(udsg) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
          
            
        #  "jet_CustomNoiseCvsL[muJet_idx]",r"Jet CustomNoise CvsL (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        #  "jet_CustomNoiseCvsB[muJet_idx]",r"Jet CustomNoise CvsB (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        #  "jet_CustomNoiseCvsL[muJet_idx]",r"Jet CustomNoise CvsL (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #  "jet_CustomNoiseCvsB[muJet_idx]",r"Jet CustomNoise CvsB (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #  "jet_CustomNoiseCvsL[0]",r"Jet CustomNoise CvsL (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
        #  "jet_CustomNoiseCvsB[0]",r"Jet CustomNoise CvsB (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      
        #  "jet_CustomNoiseBvsL[muJet_idx]",r"Jet CustomNoise BvsL (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        #  "jet_CustomNoiseBvsC[muJet_idx]",r"Jet CustomNoise BvsC (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        #  "jet_CustomNoiseBvsL[muJet_idx]",r"Jet CustomNoise BvsL (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #  "jet_CustomNoiseBvsC[muJet_idx]",r"Jet CustomNoise BvsC (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #  "jet_CustomNoiseBvsL[0]",r"Jet CustomNoise BvsL (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
        #  "jet_CustomNoiseBvsC[0]",r"Jet CustomNoise BvsC (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      
        #  "jet_CustomNoiseProb_b[muJet_idx]",r"Jet CustomNoise P(b) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        #  "jet_CustomNoiseProb_b[muJet_idx]",r"Jet CustomNoise P(b) (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #  "jet_CustomNoiseProb_b[0]",r"Jet CustomNoise P(b) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      
        #  "jet_CustomNoiseProb_bb[muJet_idx]",r"Jet CustomNoise P(bb) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        #  "jet_CustomNoiseProb_bb[muJet_idx]",r"Jet CustomNoise P(bb) (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #  "jet_CustomNoiseProb_bb[0]",r"Jet CustomNoise P(bb) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      
        #  "jet_CustomNoiseProb_c[muJet_idx]",r"Jet CustomNoise P(c) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        #  "jet_CustomNoiseProb_c[muJet_idx]",r"Jet CustomNoise P(c) (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #  "jet_CustomNoiseProb_c[0]",r"Jet CustomNoise P(c) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      
        #  "jet_CustomNoiseProb_l[muJet_idx]",r"Jet CustomNoise P(udsg) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        #  "jet_CustomNoiseProb_l[muJet_idx]",r"Jet CustomNoise P(udsg) (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #  "jet_CustomNoiseProb_l[0]",r"Jet CustomNoise P(udsg) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
            
    
        "jet_CustomFGSMCvsL[muJet_idx]",r"Jet CustomFGSM CvsL (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        "jet_CustomFGSMCvsB[muJet_idx]",r"Jet CustomFGSM CvsB (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        "jet_CustomFGSMCvsL[muJet_idx]",r"Jet CustomFGSM CvsL (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        "jet_CustomFGSMCvsB[muJet_idx]",r"Jet CustomFGSM CvsB (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        "jet_CustomFGSMCvsL[0]",r"Jet CustomFGSM CvsL (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
        "jet_CustomFGSMCvsB[0]",r"Jet CustomFGSM CvsB (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      
        "jet_CustomFGSMBvsL[muJet_idx]",r"Jet CustomFGSM BvsL (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        "jet_CustomFGSMBvsC[muJet_idx]",r"Jet CustomFGSM BvsC (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        "jet_CustomFGSMBvsL[muJet_idx]",r"Jet CustomFGSM BvsL (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        "jet_CustomFGSMBvsC[muJet_idx]",r"Jet CustomFGSM BvsC (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        "jet_CustomFGSMBvsL[0]",r"Jet CustomFGSM BvsL (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
        "jet_CustomFGSMBvsC[0]",r"Jet CustomFGSM BvsC (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
       
        "jet_CustomFGSMProb_b[muJet_idx]",r"Jet CustomFGSM P(b) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        "jet_CustomFGSMProb_b[muJet_idx]",r"Jet CustomFGSM P(b) (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        "jet_CustomFGSMProb_b[0]",r"Jet CustomFGSM P(b) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      
        "jet_CustomFGSMProb_bb[muJet_idx]",r"Jet CustomFGSM P(bb) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        "jet_CustomFGSMProb_bb[muJet_idx]",r"Jet CustomFGSM P(bb) (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        "jet_CustomFGSMProb_bb[0]",r"Jet CustomFGSM P(bb) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      
        "jet_CustomFGSMProb_c[muJet_idx]",r"Jet CustomFGSM P(c) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        "jet_CustomFGSMProb_c[muJet_idx]",r"Jet CustomFGSM P(c) (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        "jet_CustomFGSMProb_c[0]",r"Jet CustomFGSM P(c) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      
        "jet_CustomFGSMProb_l[muJet_idx]",r"Jet CustomFGSM P(udsg) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        "jet_CustomFGSMProb_l[muJet_idx]",r"Jet CustomFGSM P(udsg) (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        "jet_CustomFGSMProb_l[0]",r"Jet CustomFGSM P(udsg) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
       
    
       
        # _multi_weightingmethod_A,B,C
        
      #   "jet_Custom_A_CvsL[muJet_idx]",r"Jet Custom CvsL (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
      #   "jet_Custom_A_CvsB[muJet_idx]",r"Jet Custom CvsB (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
      #   "jet_Custom_A_CvsL[muJet_idx]",r"Jet Custom CvsL (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
      #   "jet_Custom_A_CvsB[muJet_idx]",r"Jet Custom CvsB (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
      #   "jet_Custom_A_CvsL[0]",r"Jet Custom CvsL (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      #   "jet_Custom_A_CvsB[0]",r"Jet Custom CvsB (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      #   
      #   "jet_Custom_A_BvsL[muJet_idx]",r"Jet Custom BvsL (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
      #   "jet_Custom_A_BvsC[muJet_idx]",r"Jet Custom BvsC (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
      #   "jet_Custom_A_BvsL[muJet_idx]",r"Jet Custom BvsL (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
      #   "jet_Custom_A_BvsC[muJet_idx]",r"Jet Custom BvsC (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
      #   "jet_Custom_A_BvsL[0]",r"Jet Custom BvsL (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      #   "jet_Custom_A_BvsC[0]",r"Jet Custom BvsC (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      # 
      #   "jet_Custom_A_Prob_b[muJet_idx]",r"Jet Custom P(b) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
      #   "jet_Custom_A_Prob_b[muJet_idx]",r"Jet Custom P(b) (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
      #   "jet_Custom_A_Prob_b[0]",r"Jet Custom P(b) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      # 
      #   "jet_Custom_A_Prob_bb[muJet_idx]",r"Jet Custom P(bb) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
      #   "jet_Custom_A_Prob_bb[muJet_idx]",r"Jet Custom P(bb) (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
      #   "jet_Custom_A_Prob_bb[0]",r"Jet Custom P(bb) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      # 
      #   "jet_Custom_A_Prob_c[muJet_idx]",r"Jet Custom P(c) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
      #   "jet_Custom_A_Prob_c[muJet_idx]",r"Jet Custom P(c) (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
      #   "jet_Custom_A_Prob_c[0]",r"Jet Custom P(c) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      # 
      #   "jet_Custom_A_Prob_l[muJet_idx]",r"Jet Custom P(udsg) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
      #   "jet_Custom_A_Prob_l[muJet_idx]",r"Jet Custom P(udsg) (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
      #   "jet_Custom_A_Prob_l[0]",r"Jet Custom P(udsg) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
         
         
      #   "jet_Custom_B_CvsL[muJet_idx]",r"Jet Custom CvsL (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
      #   "jet_Custom_B_CvsB[muJet_idx]",r"Jet Custom CvsB (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
      #   "jet_Custom_B_CvsL[muJet_idx]",r"Jet Custom CvsL (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
      #   "jet_Custom_B_CvsB[muJet_idx]",r"Jet Custom CvsB (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
      #   "jet_Custom_B_CvsL[0]",r"Jet Custom CvsL (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      #   "jet_Custom_B_CvsB[0]",r"Jet Custom CvsB (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      #   
      #   "jet_Custom_B_BvsL[muJet_idx]",r"Jet Custom BvsL (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
      #   "jet_Custom_B_BvsC[muJet_idx]",r"Jet Custom BvsC (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
      #   "jet_Custom_B_BvsL[muJet_idx]",r"Jet Custom BvsL (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
      #   "jet_Custom_B_BvsC[muJet_idx]",r"Jet Custom BvsC (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
      #   "jet_Custom_B_BvsL[0]",r"Jet Custom BvsL (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      #   "jet_Custom_B_BvsC[0]",r"Jet Custom BvsC (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      # 
      #   "jet_Custom_B_Prob_b[muJet_idx]",r"Jet Custom P(b) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
      #   "jet_Custom_B_Prob_b[muJet_idx]",r"Jet Custom P(b) (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
      #   "jet_Custom_B_Prob_b[0]",r"Jet Custom P(b) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      # 
      #   "jet_Custom_B_Prob_bb[muJet_idx]",r"Jet Custom P(bb) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
      #   "jet_Custom_B_Prob_bb[muJet_idx]",r"Jet Custom P(bb) (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
      #   "jet_Custom_B_Prob_bb[0]",r"Jet Custom P(bb) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      # 
      #   "jet_Custom_B_Prob_c[muJet_idx]",r"Jet Custom P(c) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
      #   "jet_Custom_B_Prob_c[muJet_idx]",r"Jet Custom P(c) (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
      #   "jet_Custom_B_Prob_c[0]",r"Jet Custom P(c) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      # 
      #   "jet_Custom_B_Prob_l[muJet_idx]",r"Jet Custom P(udsg) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
      #   "jet_Custom_B_Prob_l[muJet_idx]",r"Jet Custom P(udsg) (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
      #   "jet_Custom_B_Prob_l[0]",r"Jet Custom P(udsg) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
         
         
      #   "jet_Custom_C_CvsL[muJet_idx]",r"Jet Custom CvsL (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
      #   "jet_Custom_C_CvsB[muJet_idx]",r"Jet Custom CvsB (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
      #   "jet_Custom_C_CvsL[muJet_idx]",r"Jet Custom CvsL (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
      #   "jet_Custom_C_CvsB[muJet_idx]",r"Jet Custom CvsB (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
      #   "jet_Custom_C_CvsL[0]",r"Jet Custom CvsL (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      #   "jet_Custom_C_CvsB[0]",r"Jet Custom CvsB (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      #   
      #   "jet_Custom_C_BvsL[muJet_idx]",r"Jet Custom BvsL (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
      #   "jet_Custom_C_BvsC[muJet_idx]",r"Jet Custom BvsC (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
      #   "jet_Custom_C_BvsL[muJet_idx]",r"Jet Custom BvsL (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
      #   "jet_Custom_C_BvsC[muJet_idx]",r"Jet Custom BvsC (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
      #   "jet_Custom_C_BvsL[0]",r"Jet Custom BvsL (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      #   "jet_Custom_C_BvsC[0]",r"Jet Custom BvsC (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      # 
      #   "jet_Custom_C_Prob_b[muJet_idx]",r"Jet Custom P(b) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
      #   "jet_Custom_C_Prob_b[muJet_idx]",r"Jet Custom P(b) (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
      #   "jet_Custom_C_Prob_b[0]",r"Jet Custom P(b) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      # 
      #   "jet_Custom_C_Prob_bb[muJet_idx]",r"Jet Custom P(bb) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
      #   "jet_Custom_C_Prob_bb[muJet_idx]",r"Jet Custom P(bb) (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
      #   "jet_Custom_C_Prob_bb[0]",r"Jet Custom P(bb) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      # 
      #   "jet_Custom_C_Prob_c[muJet_idx]",r"Jet Custom P(c) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
      #   "jet_Custom_C_Prob_c[muJet_idx]",r"Jet Custom P(c) (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
      #   "jet_Custom_C_Prob_c[0]",r"Jet Custom P(c) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
      # 
      #   "jet_Custom_C_Prob_l[muJet_idx]",r"Jet Custom P(udsg) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
      #   "jet_Custom_C_Prob_l[muJet_idx]",r"Jet Custom P(udsg) (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
      #   "jet_Custom_C_Prob_l[0]",r"Jet Custom P(udsg) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
            
            
            
            
            # For soft mu bias studies
        #     "jet_CvsL[JETIDX]",r"Probe Jet DeepCSV CvsL (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #    "jet_CvsL[JETIDX]",r"Probe Jet DeepCSV CvsL (e)",NBINDISC,-0.2,1,TTSEMISELE,dataset="sele",makeROOT=True,TTSEMIWEIGHT
        #    "jet_CvsB[JETIDX]",r"Probe Jet DeepCSV CvsB (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #    "jet_CvsB[JETIDX]",r"Probe Jet DeepCSV CvsB (e)",NBINDISC,-0.2,1,TTSEMISELE,dataset="sele",makeROOT=True,TTSEMIWEIGHT

            # "jet_CvsL[JETIDX]",r"Probe Jet DeepCSV CvsL (#mu #mu)",NBINDISC,-0.2,1,TTSELMM,dataset="dmu",makeROOT=True,TTWEIGHT
            # "jet_CvsL[JETIDX]",r"Probe Jet DeepCSV CvsL (e e)",NBINDISC,-0.2,1,TTSELEE,dataset="deg",makeROOT=True,TTWEIGHT
            # "jet_CvsL[JETIDX]",r"Probe Jet DeepCSV CvsL (#mu e)",NBINDISC,-0.2,1,TTSELME,dataset="mue",makeROOT=True,TTWEIGHT
            # "jet_CvsB[JETIDX]",r"Probe Jet DeepCSV CvsB (#mu #mu)",NBINDISC,-0.2,1,TTSELMM,dataset="dmu",makeROOT=True,TTWEIGHT
            # "jet_CvsB[JETIDX]",r"Probe Jet DeepCSV CvsB (e e)",NBINDISC,-0.2,1,TTSELEE,dataset="deg",makeROOT=True,TTWEIGHT
            # "jet_CvsB[JETIDX]",r"Probe Jet DeepCSV CvsB (#mu e)",NBINDISC,-0.2,1,TTSELME,dataset="mue",makeROOT=True,TTWEIGHT
            
        #     "jet_CvsL[semitc1idx]",r"Probe Jet DeepCSV CvsL (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #    "jet_CvsL[semitc1idx]",r"Probe Jet DeepCSV CvsL (e)",NBINDISC,-0.2,1,TTSEMISELE,dataset="sele",makeROOT=True,TTSEMIWEIGHT
        #    "jet_CvsB[semitc1idx]",r"Probe Jet DeepCSV CvsB (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #    "jet_CvsB[semitc1idx]",r"Probe Jet DeepCSV CvsB (e)",NBINDISC,-0.2,1,TTSEMISELE,dataset="sele",makeROOT=True,TTSEMIWEIGHT
            
        #     "jet_CvsL[semitc1idx]",r"Probe Jet DeepCSV CvsL (#mu) (OS)",NBINDISC,-0.2,1,TTSEMISELM+"&& signWeight > 0",filePost="OS",dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #    "jet_CvsL[semitc1idx]",r"Probe Jet DeepCSV CvsL (e) (OS)",NBINDISC,-0.2,1,TTSEMISELE+"&& signWeight > 0",filePost="OS",dataset="sele",makeROOT=True,TTSEMIWEIGHT
        #    "jet_CvsB[semitc1idx]",r"Probe Jet DeepCSV CvsB (#mu) (OS)",NBINDISC,-0.2,1,TTSEMISELM+"&& signWeight > 0",filePost="OS",dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #    "jet_CvsB[semitc1idx]",r"Probe Jet DeepCSV CvsB (e) (OS)",NBINDISC,-0.2,1,TTSEMISELE+"&& signWeight > 0",filePost="OS",dataset="sele",makeROOT=True,TTSEMIWEIGHT

        #   "jet_CvsL[semitc1idx]",r"Probe Jet DeepCSV CvsL (#mu) (SS)",NBINDISC,-0.2,1,TTSEMISELM+"&& signWeight < 0",dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #   "jet_CvsL[semitc1idx]",r"Probe Jet DeepCSV CvsL (e) (SS)",NBINDISC,-0.2,1,TTSEMISELE+"&& signWeight < 0",dataset="sele",makeROOT=True,TTSEMIWEIGHT
        #   "jet_CvsB[semitc1idx]",r"Probe Jet DeepCSV CvsB (#mu) (SS)",NBINDISC,-0.2,1,TTSEMISELM+"&& signWeight < 0",dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #   "jet_CvsB[semitc1idx]",r"Probe Jet DeepCSV CvsB (e) (SS)",NBINDISC,-0.2,1,TTSEMISELE+"&& signWeight < 0",dataset="sele",makeROOT=True,TTSEMIWEIGHT
           
            
            
        #      "jet_DeepFlavCvsL[muJet_idx]",r"Jet DeepJet CvsL (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        #      "jet_DeepFlavCvsL[muJet_idx]",r"Jet DeepJet CvsL (e)",NBINDISC,-0.2,1,ESEL,dataset="sele",makeROOT=True,WCWEIGHT
        #      "jet_DeepFlavCvsB[muJet_idx]",r"Jet DeepJet CvsB (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        #      "jet_DeepFlavCvsB[muJet_idx]",r"Jet DeepJet CvsB (e)",NBINDISC,-0.2,1,ESEL,dataset="sele",makeROOT=True,WCWEIGHT
            
            #  "jet_DeepFlavCvsL[muJet_idx]",r"Jet DeepJet CvsL (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
            #  "jet_DeepFlavCvsL[muJet_idx]",r"Jet DeepJet CvsL (e)",NBINDISC,-0.2,1,TTSEMISELE,dataset="sele",makeROOT=True,TTSEMIWEIGHT
            #  "jet_DeepFlavCvsB[muJet_idx]",r"Jet DeepJet CvsB (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
            #  "jet_DeepFlavCvsB[muJet_idx]",r"Jet DeepJet CvsB (e)",NBINDISC,-0.2,1,TTSEMISELE,dataset="sele",makeROOT=True,TTSEMIWEIGHT
            
            # "jet_DeepFlavCvsL[muJet_idx]",r"Jet DeepJet CvsL (#mu #mu)",NBINDISC,-0.2,1,TTSELMM,dataset="dmu",makeROOT=True,TTWEIGHT
            # "jet_DeepFlavCvsL[muJet_idx]",r"Jet DeepJet CvsL (e e)",NBINDISC,-0.2,1,TTSELEE,dataset="deg",makeROOT=True,TTWEIGHT
            # "jet_DeepFlavCvsL[muJet_idx]",r"Jet DeepJet CvsL (#mu e)",NBINDISC,-0.2,1,TTSELME,dataset="mue",makeROOT=True,TTWEIGHT
            # "jet_DeepFlavCvsB[muJet_idx]",r"Jet DeepJet CvsB (#mu #mu)",NBINDISC,-0.2,1,TTSELMM,dataset="dmu",makeROOT=True,TTWEIGHT
            # "jet_DeepFlavCvsB[muJet_idx]",r"Jet DeepJet CvsB (e e)",NBINDISC,-0.2,1,TTSELEE,dataset="deg",makeROOT=True,TTWEIGHT
            # "jet_DeepFlavCvsB[muJet_idx]",r"Jet DeepJet CvsB (#mu e)",NBINDISC,-0.2,1,TTSELME,dataset="mue",makeROOT=True,TTWEIGHT
            
         #    "jet_DeepFlavCvsL[0]",r"Jet DeepJet CvsL (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
         #    "jet_DeepFlavCvsB[0]",r"Jet DeepJet CvsB (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
         #    "jet_DeepFlavCvsL[0]",r"Jet DeepJet CvsL (e)",NBINDISC,-0.2,1,DYSELE,dataset="deg",makeROOT=True,DYWEIGHT
         #    "jet_DeepFlavCvsB[0]",r"Jet DeepJet CvsB (e)",NBINDISC,-0.2,1,DYSELE,dataset="deg",makeROOT=True,DYWEIGHT

            # "jet_btagDeepFlavB[muJet_idx]",r"Jet DeepJet P(b) (#mu)",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
            # "jet_btagDeepFlavB[muJet_idx]",r"Jet DeepJet P(b) (#mu e)",NBINDISC,-0.2,1,TTSELME,dataset="mue",makeROOT=True,TTWEIGHT
            # "jet_btagDeepFlavB[0]",r"Jet DeepJet P(b) (#mu)",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT

            # For soft mu bias studies
            #  "jet_DeepFlavCvsL[JETIDX]",r"Probe Jet DeepJet CvsL (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
            #  "jet_DeepFlavCvsL[JETIDX]",r"Probe Jet DeepJet CvsL (e)",NBINDISC,-0.2,1,TTSEMISELE,dataset="sele",makeROOT=True,TTSEMIWEIGHT
            #  "jet_DeepFlavCvsB[JETIDX]",r"Probe Jet DeepJet CvsB (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
            #  "jet_DeepFlavCvsB[JETIDX]",r"Probe Jet DeepJet CvsB (e)",NBINDISC,-0.2,1,TTSEMISELE,dataset="sele",makeROOT=True,TTSEMIWEIGHT

            # "jet_DeepFlavCvsL[JETIDX]",r"Probe Jet DeepJet CvsL (#mu #mu)",NBINDISC,-0.2,1,TTSELMM,dataset="dmu",makeROOT=True,TTWEIGHT
            # "jet_DeepFlavCvsL[JETIDX]",r"Probe Jet DeepJet CvsL (e e)",NBINDISC,-0.2,1,TTSELEE,dataset="deg",makeROOT=True,TTWEIGHT
            # "jet_DeepFlavCvsL[JETIDX]",r"Probe Jet DeepJet CvsL (#mu e)",NBINDISC,-0.2,1,TTSELME,dataset="mue",makeROOT=True,TTWEIGHT
            # "jet_DeepFlavCvsB[JETIDX]",r"Probe Jet DeepJet CvsB (#mu #mu)",NBINDISC,-0.2,1,TTSELMM,dataset="dmu",makeROOT=True,TTWEIGHT
            # "jet_DeepFlavCvsB[JETIDX]",r"Probe Jet DeepJet CvsB (e e)",NBINDISC,-0.2,1,TTSELEE,dataset="deg",makeROOT=True,TTWEIGHT
            # "jet_DeepFlavCvsB[JETIDX]",r"Probe Jet DeepJet CvsB (#mu e)",NBINDISC,-0.2,1,TTSELME,dataset="mue",makeROOT=True,TTWEIGHT

        #    "jet_DeepFlavCvsL[semitc1idx]",r"Probe Jet DeepJet CvsL (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #    "jet_DeepFlavCvsL[semitc1idx]",r"Probe Jet DeepJet CvsL (e)",NBINDISC,-0.2,1,TTSEMISELE,dataset="sele",makeROOT=True,TTSEMIWEIGHT
        #    "jet_DeepFlavCvsB[semitc1idx]",r"Probe Jet DeepJet CvsB (#mu)",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #    "jet_DeepFlavCvsB[semitc1idx]",r"Probe Jet DeepJet CvsB (e)",NBINDISC,-0.2,1,TTSEMISELE,dataset="sele",makeROOT=True,TTSEMIWEIGHT

        #     "jet_DeepFlavCvsL[semitc1idx]",r"Probe Jet DeepJet CvsL (#mu) (OS)",NBINDISC,-0.2,1,TTSEMISELM+"&& signWeight > 0",filePost="OS",dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #    "jet_DeepFlavCvsL[semitc1idx]",r"Probe Jet DeepJet CvsL (e) (OS)",NBINDISC,-0.2,1,TTSEMISELE+"&& signWeight > 0",filePost="OS",dataset="sele",makeROOT=True,TTSEMIWEIGHT
        #    "jet_DeepFlavCvsB[semitc1idx]",r"Probe Jet DeepJet CvsB (#mu) (OS)",NBINDISC,-0.2,1,TTSEMISELM+"&& signWeight > 0",filePost="OS",dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #    "jet_DeepFlavCvsB[semitc1idx]",r"Probe Jet DeepJet CvsB (e) (OS)",NBINDISC,-0.2,1,TTSEMISELE+"&& signWeight > 0",filePost="OS",dataset="sele",makeROOT=True,TTSEMIWEIGHT

        #   "jet_DeepFlavCvsL[semitc1idx]",r"Probe Jet DeepJet CvsL (#mu) (SS)",NBINDISC,-0.2,1,TTSEMISELM+"&& signWeight < 0",dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #   "jet_DeepFlavCvsL[semitc1idx]",r"Probe Jet DeepJet CvsL (e) (SS)",NBINDISC,-0.2,1,TTSEMISELE+"&& signWeight < 0",dataset="sele",makeROOT=True,TTSEMIWEIGHT
        #   "jet_DeepFlavCvsB[semitc1idx]",r"Probe Jet DeepJet CvsB (#mu) (SS)",NBINDISC,-0.2,1,TTSEMISELM+"&& signWeight < 0",dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #   "jet_DeepFlavCvsB[semitc1idx]",r"Probe Jet DeepJet CvsB (e) (SS)",NBINDISC,-0.2,1,TTSEMISELE+"&& signWeight < 0",dataset="sele",makeROOT=True,TTSEMIWEIGHT
'''

onlyCentral = '''
        #    "jetMu_Pt",r"p^{soft #mu}_{T} [GeV] (mu)",25,0,25,MSEL,dataset="smu",WCWEIGHT
        #    "jetMu_Pt",r"p^{soft #mu}_{T} [GeV] (e)",25,0,25,ESEL,dataset="sele",WCWEIGHT
        #    "jet_Pt[muJet_idx]",r"p^{jet}_{T} [GeV] (mu)",25,20,120,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        #    "jet_Pt[muJet_idx]",r"p^{jet}_{T} [GeV] (e)",25,20,120,ESEL,dataset="sele",makeROOT=True,WCWEIGHT
        #    "jet_Eta[muJet_idx]",r"#eta_{jet} (mu)",20,-2.8,2.8,MSEL,dataset="smu",WCWEIGHT
        #    "jet_Eta[muJet_idx]",r"#eta_{jet} (e)",20,-2.8,2.8,ESEL,dataset="sele",WCWEIGHT
        #    "jet_Phi[muJet_idx]",r"#phi_{jet} (mu)",20,-3.2,3.2,MSEL,dataset="smu",WCWEIGHT
        #    "jet_Phi[muJet_idx]",r"#phi_{jet} (e)",20,-3.2,3.2,ESEL,dataset="sele",WCWEIGHT            
            
        #    "nTightMu",r"Number of tight #mu", 5,0,5,MSEL,dataset="smu",WCWEIGHT
        #    "Z_Mass_withJet","M_{#mu,jet}",40,0,120,MSEL,dataset="smu",WCWEIGHT
              
        #     "M_RelIso[0]","Rel Iso (mu)",40,0,0.08,MSEL,dataset="smu",makeROOT=True,nminus1=True,WCWEIGHT
        #     "E_RelIso[0]","Rel Iso (e)",40,0,0.08,ESEL,dataset="sele",makeROOT=True,nminus1=True,WCWEIGHT
        #     "M_dz[0]",r"M_dz (mu)",40,0,0.02,MSEL,dataset="smu",makeROOT=True,nminus1=True,WCWEIGHT
        #     "E_dz[0]",r"E_dz (e)",40,0,0.04,ESEL,dataset="sele",makeROOT=True,nminus1=True,WCWEIGHT
             "M_dxy[0]",r"M_dxy (mu)",40,0,0.004,MSEL,dataset="smu",makeROOT=True,nminus1=True,WCWEIGHT
        #     "E_dxy[0]",r"E_dxy (e)",40,0,0.02,ESEL,dataset="sele",makeROOT=True,nminus1=True,WCWEIGHT

        #     "jet_nJet",r"nJet (mu)",6,1,7,MSEL,dataset="smu",makeROOT=True,nminus1=True,WCWEIGHT
        #     "jet_nJet",r"nJet (e)",6,1,7,ESEL,dataset="sele",makeROOT=True,nminus1=True,WCWEIGHT
            # "jet_nJet",r"nJet (mu)",6,1,7,TTSEMISELM,dataset="smu",nminus1=True,TTSEMIWEIGHT
            # "jet_nJet",r"nJet (e)",6,1,7,TTSEMISELE,dataset="sele",nminus1=True,TTSEMIWEIGHT


        #    "jetMu_Pt",r"p^{soft #mu}_{T} [GeV] (mu)",25,0,25,TTSEMISELM,dataset="smu",TTSEMIWEIGHT
        #    "jetMu_Pt",r"p^{soft #mu}_{T} [GeV] (e)",25,0,25,TTSEMISELE,dataset="sele",TTSEMIWEIGHT
        #    "jet_Pt[muJet_idx]",r"p^{jet}_{T} [GeV] (mu)",25,20,120,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #    "jet_Pt[muJet_idx]",r"p^{jet}_{T} [GeV] (e)",25,20,120,TTSEMISELE,dataset="sele",makeROOT=True,TTSEMIWEIGHT
        #    "jet_Eta[muJet_idx]",r"#eta_{jet} (mu)",20,-2.8,2.8,TTSEMISELM,dataset="smu",TTSEMIWEIGHT
        #    "jet_Eta[muJet_idx]",r"#eta_{jet} (e)",20,-2.8,2.8,TTSEMISELE,dataset="sele",TTSEMIWEIGHT
        #    "jet_Phi[muJet_idx]",r"#phi_{jet} (mu)",20,-3.2,3.2,TTSEMISELM,dataset="smu",TTSEMIWEIGHT
        #    "jet_Phi[muJet_idx]",r"#phi_{jet} (e)",20,-3.2,3.2,TTSEMISELE,dataset="sele",TTSEMIWEIGHT           
              
          
          
        #     "jetMu_Pt",r"p^{soft #mu}_{T} [GeV] (#mu #mu)",25,0,25,TTSELMM,dataset="dmu",TTWEIGHT
        #     "jetMu_Pt",r"p^{soft #mu}_{T} [GeV] (e e)",25,0,25,TTSELEE,dataset="deg",TTWEIGHT
        #     "jetMu_Pt",r"p^{soft #mu}_{T} [GeV] (#mu e)",25,0,25,TTSELME,dataset="mue",TTWEIGHT
        #    "jet_Pt[muJet_idx]",r"p^{jet}_{T} [GeV] (#mu #mu)",25,20,120,TTSELMM,dataset="dmu",makeROOT=True,TTWEIGHT
        #    "jet_Pt[muJet_idx]",r"p^{jet}_{T} [GeV] (e e)",25,20,120,TTSELEE,dataset="deg",makeROOT=True,TTWEIGHT
        #    "jet_Pt[muJet_idx]",r"p^{jet}_{T} [GeV] (#mu e)",25,20,120,TTSELME,dataset="mue",makeROOT=True,TTWEIGHT
        #    "jet_Eta[muJet_idx]",r"#eta_{jet} (#mu #mu)",20,-2.8,2.8,TTSELMM,dataset="dmu",TTWEIGHT
        #    "jet_Eta[muJet_idx]",r"#eta_{jet} (e e)",20,-2.8,2.8,TTSELEE,dataset="deg",TTWEIGHT
        #    "jet_Eta[muJet_idx]",r"#eta_{jet} (#mu e)",20,-2.8,2.8,TTSELME,dataset="mue",TTWEIGHT
        #    "jet_Phi[muJet_idx]",r"#phi_{jet} (#mu #mu)",20,-3.2,3.2,TTSELMM,dataset="dmu",TTWEIGHT
        #    "jet_Phi[muJet_idx]",r"#phi_{jet} (e e)",20,-3.2,3.2,TTSELEE,dataset="deg",TTWEIGHT
        #    "jet_Phi[muJet_idx]",r"#phi_{jet} (#mu e)",20,-3.2,3.2,TTSELME,dataset="mue",TTWEIGHT      
         
         
       #     "jet_Phi[0]",r"#phi_{jet}",20,-3.2,3.2,DYSELM,dataset="dmu",DYWEIGHT
       #     "jet_Eta[0]",r"#eta_{jet}",20,-2.8,2.8,DYSELM,dataset="dmu",DYWEIGHT
       #    "jet_Pt[0]",r"p^{jet}_{T} [GeV]",25,20,120,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT              
            
        #     "jet_Phi[0]",r"#phi_{jet}",20,-3.2,3.2,DYSELE,dataset="deg",DYWEIGHT
        #     "jet_Eta[0]",r"#eta_{jet}",20,-2.8,2.8,DYSELE,dataset="deg",DYWEIGHT
        #     "jet_Pt[0]",r"p^{jet}_{T} [GeV]",25,20,120,DYSELE,dataset="deg",makeROOT=True,DYWEIGHT
              
'''

onlyKins =  '''

'''

# Jobs
cmdList = open("cmdList.txt","w")

if addsel!='': print("WARNING: YOU HAVE A CUSTOM SELECTION APPLIED!")
for systname in systs:
    global syst
    syst=systname

    if plotsysts and plot2D:
        args=[applyCuts(line.strip()) for line in arguments.split("\n") if not line.strip()=="" and not line.strip().startswith("#")]
        for i, line in enumerate(args):
            cmdList.write("Stacker.plotStack("+line.strip()+")\n")
            
    if plotBinSlices:        
        if not plotsysts and not "central" in systname: continue
        
        #varBin1=[-0.2,0.,0.2,0.4,0.6,0.8,1.]
        #varBin2=[-0.2,0.,0.2,0.4,0.6,0.8,1.]
        varBin1=[-0.2,0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]
        varBin2=[-0.2,0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]
        
        args=[line.strip() for line in plot1D.split("\n") if not line.strip()=="" and not line.strip().startswith("#")]
        
        # this adds the jobs for the stacked histograms, but without the eventual additional specifications (like normalization) to the command list
        # and if this for some reason runs after the later jobs, this overwrites the actually normalized plots
        #for i, line in enumerate(args):
        #    cmdList.write("Stacker.plotStack("+applyCuts(line).strip()+")\n")
            
        for i in range(1,len(varBin1)-1):            
            for iline, line in enumerate(args):
                deepsuff = ""
                if line.strip().startswith('"jet_DeepFlav'): deepsuff="DeepFlav"
                if line.strip().startswith('"jet_Custom'): deepsuff="Custom"
                if line.strip().startswith('"jet_CustomNoise'): deepsuff="CustomNoise"
                if line.strip().startswith('"jet_CustomFGSM'): deepsuff="CustomFGSM"
                if line.strip().startswith('"jet_Custom_A_'): deepsuff="Custom_A_"
                if line.strip().startswith('"jet_Custom_B_'): deepsuff="Custom_B_"
                if line.strip().startswith('"jet_Custom_C_'): deepsuff="Custom_C_"
                if line.strip().startswith('"jet_%sCvsL'%deepsuff): continue
                
                for sel in ["ESEL","MSEL","TTSELEE","TTSELMM","TTSELME","TTSEMISELE","TTSEMISELM"]:
                    line = line.replace(sel,sel+'+" && jet_%sCvsL[muJet_idx] >= %f && jet_%sCvsL[muJet_idx] < %f"'%(deepsuff,varBin1[i],deepsuff,varBin1[i+1]))
                for sel in ["DYSELE","DYSELM"]:
                    line = line.replace(sel,sel+'+" && jet_%sCvsL[0] >= %.2f && jet_%sCvsL[0] < %.2f"'%(deepsuff,varBin1[i],deepsuff,varBin1[i+1]))
                    
                line += ',filePost="%sCvsL_%.2f-%.2f"'%(deepsuff,varBin1[i],varBin1[i+1])
                # cmdList.write("Stacker.plotStack("+applyCuts(line).strip()+",makePNG=False)\n")
                    
        for i in range(1,len(varBin2)-1):
            for iline, line in enumerate(args):
                deepsuff = ""
                if line.strip().startswith('"jet_DeepFlav'): deepsuff="DeepFlav"
                if line.strip().startswith('"jet_Custom'): deepsuff="Custom"
                if line.strip().startswith('"jet_CustomNoise'): deepsuff="CustomNoise"
                if line.strip().startswith('"jet_CustomFGSM'): deepsuff="CustomFGSM"
                if line.strip().startswith('"jet_Custom_A_'): deepsuff="Custom_A_"
                if line.strip().startswith('"jet_Custom_B_'): deepsuff="Custom_B_"
                if line.strip().startswith('"jet_Custom_C_'): deepsuff="Custom_C_"
                if line.strip().startswith('"jet_%sCvsB'%deepsuff): continue
                
                for sel in ["ESEL","MSEL","TTSELEE","TTSELMM","TTSELME","TTSEMISELE","TTSEMISELM"]:
                    line = line.replace(sel,sel+'+" && jet_%sCvsB[muJet_idx] >= %f && jet_%sCvsB[muJet_idx] < %f"'%(deepsuff,varBin2[i],deepsuff,varBin2[i+1]))
                for sel in ["DYSELE","DYSELM"]:
                    line = line.replace(sel,sel+'+" && jet_%sCvsB[0] >= %f && jet_%sCvsB[0] < %f"'%(deepsuff,varBin2[i],deepsuff,varBin2[i+1]))
                
                line += ',filePost="%sCvsB_%.2f-%.2f"'%(deepsuff,varBin2[i],varBin2[i+1])
                cmdList.write("Stacker.plotStack("+applyCuts(line).strip()+",makePNG=False)\n")

    if "central" in systname:
        moreargs = ""
#        if validateSFs:
        moreargs+=",drawDataMCRatioLine=True"
        if normMCtodata:
            moreargs+=",normTotalMC=True"
        if plotExtra:
        
            args=[applyCuts(line.strip()) for line in onlyCentral.split("\n")+plot1D.split('\n') if not line.strip()=="" and not line.strip().startswith("#")]
            for i, line in enumerate(args):
                cmdList.write("Stacker.plotStack("+line.strip()+moreargs+")\n")

        if validateSFs:
            args=[applyCuts(line.strip()) for line in plot1D.split('\n') if not line.strip()=="" and not line.strip().startswith("#")]
            for i, line in enumerate(args):
                if "DeepFlavCvs" in line or "btagDeepFlav" in line: SFfiles = SFfilesDeepJet
                else: SFfiles = SFfilesDeepCSV
                for SF in SFfiles:
                    for histsuff in SFhistSuff:
                        if SF == "" and histsuff != "": continue
                        cmdList.write("Stacker.plotStack(%s,SFfile=\"%s\",SFhistSuff=\"%s\",drawDataMCRatioLine=True)\n"%(line.strip(),SF,histsuff))
        else:
            args=[applyCuts(line.strip()) for line in plot1D.split('\n') if not line.strip()=="" and not line.strip().startswith("#")]
            #print(args)
            for i, line in enumerate(args):
                #pass
                cmdList.write("Stacker.plotStack(%s)\n"%(line.strip()+moreargs))

        #if plotBinnedKins:
            #varBin1=[-0.2,0.,0.2,0.4,0.6,0.8,1.]
            #varBin2=[-0.2,0.,0.2,0.4,0.6,0.8,1.]
            ##varBin1=[-0.2,0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]
            ##varBin2=[-0.2,0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]

            #for i in range(len(varBin1)-1):
                #for j in range(len(varBin2)-1):
                    #args=[line.strip() for line in onlyCentral.split("\n") if not line.strip()=="" and not line.strip().startswith("#")]
                    #for iline, line in enumerate(args):
                        #for sel in ["ESEL","MSEL","TTSELEE","TTSELMM","TTSELME","TTSEMISELE","TTSEMISELM"]:
                            #line = line.replace(sel,sel+'+[["jet_CvsL[muJet_idx]"],["jet_CvsB[muJet_idx]"]]')
                        #line = line.replace("DYSELE",'DYSELE+[["jet_CvsL",0],["jet_CvsB[0]"]]').replace("DYSELM",'DYSELM+[["jet_CvsL",0],["jet_CvsB[0]"]]')
                        #for cut in ["ECUT","MCUT","TTCUTEE","TTCUTMM","TTCUTME","TTSEMICUTM","TTSEMICUTE","DYCUTE","DYCUTM"]:
                            #line = line.replace(cut,cut+'+[['+str(varBin1[i])+','+str(varBin1[i+1])+'],['+str(varBin2[j])+','+str(varBin2[j+1])+']]')
                        #cmdList.write("Stacker.plotStack("+applyCuts(line).strip()+")\n")    

cmdList.close()
