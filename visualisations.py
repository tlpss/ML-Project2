from matplotlib import colors
import numpy as np 
import matplotlib.pyplot as plt
import json
import os

from stochastic_models import MaxCallStochasticModel

def visualise_stock_prices(S,DeltaT,start_time = 0):
    """
    Plots stock prices 

    :param S: Stock prices of a bundle of stocks
    :type S: d x T np 2D array
    :param DeltaT: array of evaluation moments for stock prices
    :type DeltaT: list of len T-1
    :param start_time: time relative to which the DeltaT array times are expressed, defaults to 0
    :type start_time: int, optional
    """
    d,T = S.shape
    time = np.ones(T)*start_time
    for i in range(1,len(time)):
        time[i] = time[i-1] + DeltaT[i-1]
    plt.figure()

    for index in range(S.shape[0]): 
        plt.plot(time,S[index,:],"-o", label=f"stock {index}",alpha=0.5)
    
    plt.scatter(time[-1],1,marker="x",color="black")
    plt.scatter(time[-1],np.max(S[:,-1]),marker="x",color="black")
    plt.vlines( time[-1] ,1,np.max(S[:,-1]),label = "MaxCall value ",linestyle='dashed',color="black",alpha=0.8)
    
    plt.xlabel("time")
    plt.ylabel("stock value")
    plt.legend()
    plt.show()


def visualise_json_grid_files(file_names, M_grid,alpha_grid,ref_error, title):

    means = np.ones((len(alpha_grid),len(M_grid)))
    sigmas = np.ones((len(alpha_grid),len(M_grid)))
    # open files, extract the result array
    for filename in file_names:

        with open(filename,"r") as f:
            data = f.read()
            data = json.loads(data)
            for run in data["errors"]:
                m,alpha, results  = run[0],run[1],run[2]
                run_errors = np.array(results)
                means[alpha_grid.index(alpha),M_grid.index(m)] = run_errors.mean()
                sigmas[alpha_grid.index(alpha),M_grid.index(m)] = run_errors.std()
    



    plt.hlines(ref_error,xmin=M_grid[0],xmax=M_grid[-1],linestyles='dashed',color="black",label="reference error")
    for i in range(len(alpha_grid)):
        plt.errorbar(np.array(M_grid),means[i,:],sigmas[i,:],marker ='o',label = f"alpha = {alpha_grid[i]}")
    #plt.title(title)
    plt.xlabel("M")
    plt.xticks(M_grid)
    plt.ylabel("normalized error")
    plt.ylim(0.11,0.16)
    plt.legend(loc='upper right')
    plt.show()

def vis_aggregating_comparsion():
    N_train = 5000
    N_test  = 50000
    d = 1 
    DeltaT = [1/12,11/12]
    M_grid  = [1,3,5,7,9]
    alpha_grid = [0.2,0.3,0.4,0.5]
    hard_bagging  = np.array([[[0.18405687, 0.19855485, 0.2017712 ],
        [0.18445979, 0.19968908, 0.19287461],
        [0.15620511, 0.16670611, 0.17330706],
        [0.23161205, 0.25864762, 0.25397623]],

       [[0.16176284, 0.16999539, 0.16423112],
        [0.11040252, 0.11271713, 0.12160244],
        [0.13004539, 0.1495795 , 0.14888737],
        [0.12220362, 0.12685371, 0.13711206]],

       [[0.12117313, 0.12795464, 0.12356273],
        [0.11650846, 0.13083649, 0.13522609],
        [0.12868414, 0.14730176, 0.15341002],
        [0.17301261, 0.17972655, 0.18560729]],

       [[0.13359486, 0.14773783, 0.1403787 ],
        [0.12992137, 0.14699154, 0.15101376],
        [0.13340449, 0.15066672, 0.15609818],
        [0.15425925, 0.16840871, 0.17713583]],

       [[0.12085441, 0.1336169 , 0.12740177],
        [0.11574074, 0.12776819, 0.13063216],
        [0.11985359, 0.13229665, 0.13863741],
        [0.13764079, 0.14461473, 0.15828688]]])

    hard_pasting = np.array([[[0.17313162, 0.16497799, 0.1502658 ],
        [0.14230661, 0.14649089, 0.14420284],
        [0.16531526, 0.1554516 , 0.16018578],
        [0.17596006, 0.19948613, 0.19928969]],

       [[0.13642943, 0.15689   , 0.14867949],
        [0.13062819, 0.15536314, 0.1563323 ],
        [0.12925257, 0.13830215, 0.15658628],
        [0.17148612, 0.18341087, 0.19139432]],

       [[0.13380715, 0.14186233, 0.13301586],
        [0.13645938, 0.13942144, 0.14218394],
        [0.12057771, 0.13042472, 0.14003114],
        [0.14838035, 0.16368196, 0.17362762]],

       [[0.13368824, 0.14784109, 0.14047682],
        [0.13001218, 0.14709428, 0.15111931],
        [0.12574064, 0.14145635, 0.15155438],
        [0.15436707, 0.16852642, 0.17725964]],

       [[0.13499191, 0.14808594, 0.14046566],
        [0.12420675, 0.13831837, 0.14261537],
        [0.11861174, 0.12642599, 0.13622762],
        [0.14179389, 0.15177186, 0.16283129]]])

    soft_bagging = np.array([[[0.22261526679701296, 0.23463680268837922, 0.21787666659850946], [0.16273217733760342, 0.1763960349201307, 0.15458612463101926], [0.1846541623022965, 0.2133472214574922, 0.21778405575238682], [0.14248851405655863, 0.15918576321666902, 0.16660779915500862]], 
        [[0.11276971524447511, 0.12682011638515028, 0.12119919154019855], [0.13742421125355048, 0.1535012304211044, 0.1504570079709425], [0.13962986134032215, 0.1521038811668313, 0.16111615879649366], [0.12508112942216215, 0.1336038275630905, 0.13945385126362697]],
        [[0.11198646612399402, 0.12294716754581123, 0.11879628771633076], [0.11808861220556148, 0.13294500568817935, 0.1319737811268167], [0.11865641565464927, 0.12698947586659123, 0.13668114132815679], [0.1281885760531762, 0.13915758861289138, 0.15057387335460853]], 
        [[0.114766542844548, 0.12810808684132308, 0.12259351007575611], [0.12327316590711314, 0.13761057105947, 0.142858793024612], [0.11633595108581112, 0.12487340456954936, 0.1359011762331626], [0.14389293105796894, 0.1505961132686077, 0.16396932512794962]],
        [[0.10949335990152097, 0.1227084300356834, 0.1190495400137878], [0.11255631818389407, 0.12714956812834693, 0.13204309625076707], [0.14071261141243815, 0.1515791084635628, 0.1573743227587902], [0.1556573156335792, 0.16631247695810955, 0.17611937656692528]]])


    soft_bagging_mean = soft_bagging.mean(axis=2)
    soft_bagging_sigmas = soft_bagging.std(axis=2)

    hard_bagging_mean = hard_bagging.mean(axis=2)
    hard_bagging_sigmas = hard_bagging.std(axis=2)

    hard_pasting_mean = hard_pasting.mean(axis=2)
    hard_pasting_sigmas = hard_pasting.std(axis=2)

    plt.hlines(0.1419,xmin=M_grid[0],xmax=M_grid[-1],linestyles='dashed',label="reference error",color="black")

    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    print(colors)
    for i in range(len(alpha_grid)):
        plt.errorbar(np.array(M_grid),soft_bagging_mean[:,i],soft_bagging_sigmas[:,i],marker ='o',linestyle="-",color= colors[0],alpha=0.2)
        plt.errorbar(np.array(M_grid),hard_bagging_mean[:,i],hard_bagging_sigmas[:,i],marker ='s',linestyle="-",color= colors[1],alpha=0.2)
        plt.errorbar(np.array(M_grid),hard_pasting_mean[:,i],hard_pasting_sigmas[:,i],marker ='D',linestyle="-",color =colors[2],alpha=0.2)

    # plot best grid values with higher alpha & label
    i_soft = 0
    i_hard_bag = 1
    i_hard_paste = 2

    plt.errorbar(np.array(M_grid),hard_pasting_mean[:,i_hard_paste],hard_pasting_sigmas[:,i_hard_paste],marker ='D',linestyle="-",color =colors[2],label = f"hard pasting, alpha= {alpha_grid[i_hard_paste]}",alpha=0.9)
    plt.errorbar(np.array(M_grid),hard_bagging_mean[:,i_hard_bag],hard_bagging_sigmas[:,i_hard_bag],marker ='s',linestyle="-",color= colors[1],label = f"hard bagging, alpha= {alpha_grid[i_hard_bag]}",alpha=0.9)
    plt.errorbar(np.array(M_grid),soft_bagging_mean[:,i_soft],soft_bagging_sigmas[:,i_soft],marker ='o',linestyle="-",color= colors[0],label = f"soft bagging, alpha= {alpha_grid[i_soft]}",alpha=0.9)
    
    #plt.title(f"Comparison of different aggregating methods: N_train = {N_train}, d= {d}, N_test= {N_test}")
    plt.xlabel("M")
    plt.xticks(M_grid)
    plt.ylabel("normalized error")
    plt.legend()
    plt.show()

def vis_deregularized_soft_bagging_comparison():
    M_grid  = [1,3,5,7,9]
    alpha_grid = [0.2,0.3,0.4,0.5]
    lambda_dereg = 1e-4
    lambda_opt = 2.5e-6
    soft_bagging = np.array([[[0.22261526679701296, 0.23463680268837922, 0.21787666659850946], [0.16273217733760342, 0.1763960349201307, 0.15458612463101926], [0.1846541623022965, 0.2133472214574922, 0.21778405575238682], [0.14248851405655863, 0.15918576321666902, 0.16660779915500862]], 
        [[0.11276971524447511, 0.12682011638515028, 0.12119919154019855], [0.13742421125355048, 0.1535012304211044, 0.1504570079709425], [0.13962986134032215, 0.1521038811668313, 0.16111615879649366], [0.12508112942216215, 0.1336038275630905, 0.13945385126362697]],
        [[0.11198646612399402, 0.12294716754581123, 0.11879628771633076], [0.11808861220556148, 0.13294500568817935, 0.1319737811268167], [0.11865641565464927, 0.12698947586659123, 0.13668114132815679], [0.1281885760531762, 0.13915758861289138, 0.15057387335460853]], 
        [[0.114766542844548, 0.12810808684132308, 0.12259351007575611], [0.12327316590711314, 0.13761057105947, 0.142858793024612], [0.11633595108581112, 0.12487340456954936, 0.1359011762331626], [0.14389293105796894, 0.1505961132686077, 0.16396932512794962]],
        [[0.10949335990152097, 0.1227084300356834, 0.1190495400137878], [0.11255631818389407, 0.12714956812834693, 0.13204309625076707], [0.14071261141243815, 0.1515791084635628, 0.1573743227587902], [0.1556573156335792, 0.16631247695810955, 0.17611937656692528]]])


    soft_bagging_mean = soft_bagging.mean(axis=2)
    soft_bagging_sigmas = soft_bagging.std(axis=2)

    dereg_soft_bagging = np.array([[[0.2170534848611092, 0.23222713932240646, 0.22408812437158748], [0.19678802380872829, 0.18922176814121275, 0.1665407920044947], [0.13656316970519256, 0.1634964361054732, 0.15881507714176724], [0.12856827086596193, 0.1422600967149596, 0.13466856943863628]], 
    [[0.1315914921032083, 0.1358846005386649, 0.13863022973155878], [0.1019860623938885, 0.10139735059398657, 0.09691510017659034], [0.09456578107754722, 0.10924610608054819, 0.1146119542544666], [0.1153723018632555, 0.12280786295960323, 0.1335160882726738]],
     [[0.11534792739264185, 0.11846623582394787, 0.11833288246468494], [0.10381062576658612, 0.1066346656013585, 0.09674675772691814], [0.10309323370316904, 0.11719487891090585, 0.12018586390348397], [0.10880820460806287, 0.11582760360686284, 0.1250881216242166]],
      [[0.16055826042353466, 0.17092006904240464, 0.1730121983181031], [0.0960602628497013, 0.09322749372139459, 0.08523263902108667], [0.10381595372667494, 0.11791827431052168, 0.12011400866027899], [0.11666093247386633, 0.12876371483235613, 0.13665557118055022]],
     [[0.13363174656999655, 0.1411245286795573, 0.14040870987256274], [0.09033271937911595, 0.09250307591000216, 0.08572064777449828], [0.10238922502711323, 0.11382737536411948, 0.11783578006333996], [0.10657295172212007, 0.1170313210055382, 0.126085683841818]]])

    dereg_soft_bagging_mean = dereg_soft_bagging.mean(axis=2)
    dereg_soft_bagging_sigmas = dereg_soft_bagging.std(axis=2)

    plt.hlines(0.1419,xmin=M_grid[0],xmax=M_grid[-1],linestyles='dashed',label="reference error",color="black")

    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for i in [0,1,2]:
        plt.errorbar(np.array(M_grid),soft_bagging_mean[:,i],soft_bagging_sigmas[:,i],marker ='o',linestyle="dotted",color= colors[i],alpha=0.8, label=f"lambda = 2.6e-6 (optimal), alpha = {alpha_grid[i]}")
    for i in [1,2]:
        plt.errorbar(np.array(M_grid),dereg_soft_bagging_mean[:,i],dereg_soft_bagging_sigmas[:,i],marker ='s',linestyle="-",color= colors[i],alpha=0.8,label=f"lambda = 1e-5 , alpha = {alpha_grid[i]}")

    #plt.title(f"Comparison of different aggregating methods: N_train = {N_train}, d= {d}, N_test= {N_test}")
    plt.xlabel("M")
    plt.xticks(M_grid)
    plt.ylabel("normalized error")
    plt.legend()
    plt.show()

def vis_scitas_hard_bagging():
    M_grid  = [1,4,7,10,13,16,19]

    hard_bagging = [[1, 0.6, [0.1549997293851802, 0.15375185189475996]], [4, 0.6, [0.13012101166021306, 0.1291694128341514]], [7, 0.6, [0.12584157170201724, 0.12516689668218584]], [10, 0.6, [0.1252757922730144, 0.12465275819187449]], [13, 0.6, [0.12377716679882575, 0.12326541250641943]], [16, 0.6, [0.12445440478488304, 0.12364453281812814]], [19, 0.6, [0.12323222749470876, 0.12264874917436597]]]

    hard_bagging = np.array([element[2] for element in hard_bagging])


    hard_bagging_mean = np.array(hard_bagging).mean(axis=1)
    hard_bagging_sigmas = np.array(hard_bagging).std(axis=1)
    print(hard_bagging_mean)
    plt.hlines(0.124,xmin=M_grid[0],xmax=M_grid[-1],linestyles='dashed',label="reference error",color="black")
    plt.errorbar(np.array(M_grid),hard_bagging_mean[:],hard_bagging_sigmas[:],marker ='s',linestyle="-",label = f"hard bagging, alpha= 0.6",alpha=0.9)
    plt.xlabel("M")
    plt.xticks(M_grid)
    plt.ylim(0.11,0.16)
    plt.ylabel("normalized error")
    plt.legend()
    plt.show()

#### actual vis

def visualise_stocks():
    s = MaxCallStochasticModel(1,6,[1/12,11/12])
    s.generate_samples()
    visualise_stock_prices(s.S[0],s.delta_T)

def vis_soft_bagging_scitas():
    filenames = ["SCITAS-results\mpi_bagging2020-12-09.15-41-43.json","SCITAS-results\mpi_bagging2020-12-09.19-25-34.json",
    "SCITAS-results\mpi_bagging2020-12-09.21-04-17.json","SCITAS-results\mpi_bagging2020-12-10.00-05-00.json",
    "SCITAS-results\mpi_bagging2020-12-10.16-00-54.json"]

    visualise_json_grid_files(filenames,[1,4,7,10,13,16,19],[0.3,0.4,0.5,0.6,0.7],.124,"Soft Bagging Normalized error: N_train = 20 000, d= 6, N_test= 100 000")

def vis_extended_soft_bagging_scitas():
    """
    run for 0.6 with M going to 28
    """
    filenames = ["SCITAS-results\mpi_bagging2020-12-10.03-56-54.json"]
    visualise_json_grid_files(filenames,[1,4,7,10,13,16,19,22,25,28],[0.6],.124,"Soft Bagging Normalized error: N_train = 20 000, d= 6, N_test= 100 000")


if __name__ == "__main__":
    #visualise_stocks()
    #vis_soft_bagging_scitas()
    #vis_extended_soft_bagging_scitas()
    #vis_aggregating_comparsion()
    #vis_deregularized_soft_bagging_comparison()
    vis_scitas_hard_bagging()