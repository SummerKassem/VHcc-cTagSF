import os, subprocess, sys
condordir="condor/"
if len(sys.argv) > 1: condid=sys.argv[1]+"*"
else: condid = ""

userinp = raw_input("Split jobs by JECs? (y/n): ")
if userinp.strip().lower()=='y':
    split=True
    print "Will split by JECs."
else:
    split=False
    print "Will NOT split by JECs."

os.system("grep -Ril %s/*%s.log -e aborted &> failedlist.txt"%(condordir,condid))
os.system("mkdir -p %s/oldlogs"%condordir)
failednums = []
with open("failedlist.txt",'r') as lst:
    for line in lst:
        failednums.append(line.split("-")[1].split(".")[1])
failednums.sort()
print failednums
missed = failednums[:]
#concstr = " ".join(failednums)
#print concstr   
os.system("mv cmdList_Wc_PFNano.txt cmdList_Wc_PFNano_old.txt")

with open("cmdList_Wc_PFNano.txt","w") as fl:
    f=open('cmdList_Wc_PFNano_old.txt','r')
    ln=f.readlines()
    for failed in failednums:
        if not split:
            fl.write(ln[int(failed)])
        else:
            for ijec in range(5):
                fl.write(ln[int(failed)].strip()+" "+str(ijec)+'\n')
                if "Single" in ln[int(failed)] or "Double" in ln[int(failed)] or "EGamma" in ln[int(failed)] or "MuonEG" in ln[int(failed)]: break
        os.system("mv %s/*%s.%s.log %s/oldlogs"%(condordir,condid,failed,condordir))

if split:
    print "Resubmit with `condor_submit submit_split.sub`."
    print "Remember to edit the output directory in condor_runscript_common_splitJEC.sh to make it the same as in condor_runscript_common.sh."
