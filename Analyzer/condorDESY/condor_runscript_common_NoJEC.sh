#!/bin/bash
    
    # OLD! Custom ~DeepCSV
    # adjust the weighting method, look up the definitions in customTaggerInference.py
    WM="_200"  # example for single weighting method alone (using raw/Noise/FGSM inputs)
#    WM="_multi_adv_tr_eps0.01_5,10,100"  # example for three epochs of one weighting method

    # NEW! Custom ~DeepJet
    #WM="_nominal_best"
#    WM="_multi_nominal_5,15,30"

    export OUTPUTDIR=/nfs/dust/cms/user/summer/ctag_condor/2022_June_2017_$4${WM}/
        OUTPUTNAME=outTree.root

        CONDOR_CLUSTER_ID=$1
        CONDOR_PROCESS_ID=$2
        INPFILE=$3
        TEMP=$5
    
        if  [[ $4 == "Wc" ]]; then
            PYFILE="WcSelection_new.py"
        elif  [[ $4 == "DY" ]]; then
            PYFILE="DYJetSelection_new.py"
        elif  [[ $4 == "TT" ]]; then
            PYFILE="TTbSelection.py"
        elif  [[ $4 == "TTNoMu" ]]; then
            PYFILE="TTbNoMuSelection.py"
        elif  [[ $4 == "WcNoMu" ]]; then
            PYFILE="WcNoMuSelection.py"
        fi
        
        export PATH=/afs/desy.de/common/passwd:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin:/cvmfs/grid.cern.ch/emi3ui-latest/bin:/cvmfs/grid.cern.ch/emi3ui-latest/sbin:/cvmfs/grid.cern.ch/emi3ui-latest/usr/bin:/cvmfs/grid.cern.ch/emi3ui-latest/usr/sbin:$PATH

        if [ $TEMP == 'yes' ]
        then 
            echo "copy scripts to tmp and cd to tmp"
            tmp_dir=$(mktemp -d)
            cp -r /nfs/dust/cms/user/summer/VHcc-cTagSF/Analyzer/${PYFILE} /nfs/dust/cms/user/summer/VHcc-cTagSF/Analyzer/condorDESY/customTaggerInference.py /nfs/dust/cms/user/summer/VHcc-cTagSF/Analyzer/condorDESY/focal_loss.py /nfs/dust/cms/user/summer/VHcc-cTagSF/Analyzer/nuSolutions.py /nfs/dust/cms/user/summer/VHcc-cTagSF/Analyzer/scalefactors* $tmp_dir
            cd $tmp_dir
        else
            echo "copy scripts to scratch"
            cp -r /nfs/dust/cms/user/summer/VHcc-cTagSF/Analyzer/${PYFILE} /nfs/dust/cms/user/summer/VHcc-cTagSF/Analyzer/condorDESY/customTaggerInference.py /nfs/dust/cms/user/summer/VHcc-cTagSF/Analyzer/condorDESY/focal_loss.py /nfs/dust/cms/user/summer/VHcc-cTagSF/Analyzer/nuSolutions.py /nfs/dust/cms/user/summer/VHcc-cTagSF/Analyzer/scalefactors* $_CONDOR_SCRATCH_DIR
        fi
        
        echo "setting up the grid commands"
        source /cvmfs/grid.cern.ch/centos7-umd4-ui-4_200423/etc/profile.d/setup-c7-ui-example.sh
        
        #echo "    which xrdcp"
        #which xrdcp
        
        echo "set up proxy"
        if [ -f "x509up_u52336" ]; then
            export X509_USER_PROXY=x509up_u52336
        fi
        
        echo "    voms-proxy-info -all"
        voms-proxy-info -all
        
        INPPREFIX="root://grid-cms-xrootd.physik.rwth-aachen.de:1094/"
        
        echo "copy actual input file"
        xrdcp ${INPPREFIX}${INPFILE} ./infile.root

        #echo "    echo PATH:"
        #echo $PATH
        #echo "    content of pwd"
        #ls
        #echo "    which python3"
        #which python3
        
        ENVNAME=deepCSV_env
        ENVDIR=$ENVNAME

        export PATH
        #echo "    echo PATH:"
        #echo $PATH
        mkdir $ENVDIR
        echo "setup conda"
        tar -xzf /nfs/dust/cms/user/summer/${ENVNAME}.tar.gz -C ${ENVDIR}

        source ${ENVNAME}/bin/activate
        #echo "    which python3"
        #which python3
        conda-unpack
        #which python3
        #echo "    echo PATH:"
        #echo $PATH
        echo "start with custom tagger"
        python3 customTaggerInference.py ${INPPREFIX}${INPFILE} ${WM} ${OUTPUTDIR}

        echo "setting up the environment (CMSSW)"
        cd /cvmfs/cms.cern.ch/slc6_amd64_gcc630/cms/cmssw/CMSSW_10_2_0_pre6/src
        source /cvmfs/cms.cern.ch/cmsset_default.sh
        #which xrdcp
        eval `scramv1 runtime -sh`

        #echo "    echo PATH:"
        #echo $PATH

        source /cvmfs/grid.cern.ch/etc/profile.d/setup-cvmfs-ui.sh
        
        if [ $TEMP == 'yes' ]
        then
            echo "changing to tmp dir again"
            cd $tmp_dir
        else
            echo "changing to scratch dir again"
            cd $_CONDOR_SCRATCH_DIR
        fi

        #echo "    pwd and ls"
        #pwd
        #ls 

        
        #path_without_pref=${INPFILE#"$INPPREFIX"}
        echo "running python script (Analyzer)"
        python ${PYFILE} ${INPFILE}

        # the variable '$?' holds the exit code of the most recently executed command
        rc=$?
        if [[ $rc == 99 ]]               
        then  
            echo "Output file already exists. Aborting job with exit code 0." $rc  
            exit 0       
        fi
        if [[ $rc != 0 ]]
        then
            echo "got exit code from python code: " $rc
            exit $rc
        fi
        echo "done running, now copying output to DUST"

        echo "copying output"
        SAMPNAME=$(bash dirName.sh)
        FLNAME=$(bash flName.sh)
        mkdir -p ${OUTPUTDIR}${SAMPNAME}
        until cp -vf ${OUTPUTNAME} ${OUTPUTDIR}${SAMPNAME}"/outTree_"${FLNAME}".root"; do
            echo "copying output failed. Retrying..."
            sleep 60
        done
        echo "copied output successfully"
        
        echo "Clean up after yourself"
        
        if [ $TEMP == 'yes' ]
        then
            cd $TMP
            rm -r $tmp_dir
        else
            rm x509up_u52336
            rm *.root *.py *.cc *.npy
            rm -r ./${ENVDIR} ./scalefactors*
        fi
        

        echo "all done!"
