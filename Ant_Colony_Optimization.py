import random as rand
import math as mt
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


def Compute_Probabilities(Weight_Array):              # (1)
    Total = sum(Weight_Array)
    Choosen_Probabilities = [Weight / Total for Weight in Weight_Array]
    return Choosen_Probabilities


def Compute_Weight(rank_j, np, Sigma):                              # (2)
    Expo = -((rank_j - 1)**2) / (2 * Sigma**2 * np**2)
    Weight = (1 / (Sigma * np * mt.sqrt(2 * mt.pi))) * mt.exp(Expo)
    return Weight

def Gaussian_Distr(x_d, mu_d, Delta_d):                      # (3)
    Expo = -((x_d - mu_d)**2) / (2 * Delta_d**2)
    GaussValue = (1 / (Delta_d * mt.sqrt(2 * mt.pi))) * mt.exp(Expo)
    return GaussValue

def Compute_Delta(X_D_Val, x_j_d, Xi, np):                     # (4)
    sum_of_diffs = sum(abs(x_i_d - x_j_d) for x_i_d in X_D_Val)
    Delta_d = Xi * sum_of_diffs / (np - 1)
    return Delta_d

def Compute_Sigma(FS_max_i, FS_min_i, FS_max, FS_min, Eta):       # (5)
    Expo = -(FS_max_i - FS_min_i) / (FS_max - FS_min + Eta)
    sigma_i = 0.1 + 0.3 * mt.exp(Expo)
    return sigma_i

def Compute_Mu(x_j_d, x_seed_d, DE_Mutator):                      # (6)
    return x_j_d + DE_Mutator * (x_seed_d - x_j_d)

def SingleModal_P_i(FSE_i, FSE_max):                                    # (7)
    P_i = FSE_i / FSE_max
    return P_i

def MultiModal_P_i(FSE_i, FSE_min, FSE_max, Eta):                   # (8)
    numerator = FSE_i + abs(FSE_min) + Eta
    denominator = FSE_max + abs(FSE_min) + Eta
    P_i = numerator / denominator
    return P_i

def Compute_Metrics(NPF, NKP, NR, NSR, FE):                         # (9)
    PR = sum(NPF) / (NKP * NR)
    SR = NSR / NR
    CS = sum(FE) / NR
    return PR, SR, CS


def calDis(individual, reference_point):
    total = 0
    for i in range(len(individual)):
        total += (individual[i] - reference_point[i]) ** 2
    return mt.sqrt(total)


def Algorithm_1(popArray, clusSize):
    clusters = []
    indexItr = []
    minDis=0
    neatInd=-1
    while len(indexItr) + clusSize <= len(popArray):
        refPoint = [rand.random() for _ in popArray[0]]
        minDis = float('inf')
        nearInd = -1
        
        for i in range(len(popArray)):
            if i in indexItr:
                continue
            dis = calDis(popArray[i], refPoint)
            if dis < minDis:
                minDis = dis
                nearInd = i
                
        indNear = popArray[nearInd]
        indexItr.append(nearInd)
        cluster = [indNear]
        
        disArray = []
        for i in range(len(popArray)):
            if i in indexItr:
                continue
            dis = calDis(popArray[i], indNear)
            disArray.append((dis, i))
        
        disArray.sort()
        
        for j in range(clusSize - 1):
            if j < len(disArray):
                index = disArray[j][1]
                cluster.append(popArray[index])
                indexItr.append(index)
        
        elemPop = []
        for i in range(len(popArray)):
            if i not in indexItr:
                elemPop.append(popArray[i])
        popArray = elemPop
        
        clusters.append(cluster)
    
    return clusters


def combinePopAndFit(comPopFit, popArray, fitnessVal):
    for i in range(len(popArray)):
        comPopFit.append((popArray[i], fitnessVal[i]))  

def sortCombineArray(comPopFit):
    for i in range(len(comPopFit)):
        for j in range(i + 1, len(comPopFit)):
            if comPopFit[i][1] < comPopFit[j][1]:
                temp = comPopFit[i]
                comPopFit[i] = comPopFit[j]
                comPopFit[j] = temp

def Algorithm_2(popArray, fitnessVal, clusSize):
    specieArray = []
    comPopFit = []

    while len(popArray) >= clusSize:
        combinePopAndFit(comPopFit, popArray, fitnessVal)
        sortCombineArray(comPopFit)
        
        popArray = []
        fitnessVal = []
        for x in comPopFit:
            popArray.append(x[0])
            fitnessVal.append(x[1])

        bestInd = popArray[0]
        specieCluster = [bestInd]

        disArray = []
        for i in range(1, len(popArray)):
            dis = calDis(popArray[i], bestInd)
            disArray.append((dis, i))

        disArray.sort()

        for j in range(clusSize - 1):
            if j < len(disArray):
                index = disArray[j][1]
                specieCluster.append(popArray[index])

        elemPop = []
        elemFit = []
        for i in range(len(popArray)):
            if popArray[i] not in specieCluster:
                elemPop.append(popArray[i])
                elemFit.append(fitnessVal[i])

        popArray = elemPop
        fitnessVal = elemFit

        specieArray.append(specieCluster)

    return specieArray

def Algorithm_4(NS, Niches, FS_max, FS_min, Eta, F, Xi, DE_Mutator):
    Fitted_Solutions = []

    Idx = 0
    while Idx < len(Niches):  # Step 1
        LNiche = Niches[Idx]
        FS_max_i = LNiche['FS_max_i']  # 1.1
        FS_min_i = LNiche['FS_min_i']  # 1.1
        Niche_Solutions = LNiche['solutions']

        Sigma = Compute_Sigma(FS_max_i, FS_min_i, FS_max, FS_min, Eta)  # 1.2

        Weight_Array = []
        Rank = 0
        while Rank < len(Niche_Solutions):
            Weight = Compute_Weight(Rank + 1, len(Niche_Solutions), Sigma)
            Weight_Array.append(Weight)  # 1.3
            Rank += 1

        Choosen_Probabilities = Compute_Probabilities(Weight_Array)  # 1.3

        ns_count = 0
        while ns_count < NS:  # 1.4
            Idx_Chosen = rand.choices( population=list(range(len(Niche_Solutions))), weights=Choosen_Probabilities, k=1 )[0]
            Chosen_Sol = Niche_Solutions[Idx_Chosen]

            if rand.random() <= 0.5:  # 1.4.2
                Updated_Sol = Chosen_Sol[:]
            else:
                Seed_Count = [i for i in range(len(Niche_Solutions)) if i != Idx_Chosen]
                Seed_Idx = rand.choice(Seed_Count)
                Seed_Sol = Niche_Solutions[Seed_Idx]
                Updated_Sol = []
                j = 0
                while j < len(Chosen_Sol):
                    mu_d = Compute_Mu(Chosen_Sol[j], Seed_Sol[j], DE_Mutator)
                    Updated_Sol.append(mu_d)
                    j += 1

            Solution = []
            j = 0
            while j < len(Updated_Sol):
                X_D_Val = [Sol[j] for Sol in Niche_Solutions]
                Delta_d = Compute_Delta(X_D_Val, Chosen_Sol[j], Xi, NS)  # 1.4.3
                GaussValue = Gaussian_Distr(Chosen_Sol[j], Updated_Sol[j], Delta_d)  # 1.4.4
                Solution.append(GaussValue)
                j += 1

            Fitness = F(Solution[0])
            Fitted_Solutions.append((Solution, Fitness))
            ns_count += 1

        Idx += 1

    return Fitted_Solutions


def Algorithm_5(S, FSE, Delta, N, Eta):
    FSE_min = min(FSE)                  # Step 1
    FSE_max = max(FSE)                  # Step 1
    Flag = False                        # Step 1

    if FSE_min <= 0:                                # Step 2
        FSE_max = FSE_max + abs(FSE_min) + Eta      # Step 2
        Flag = True                                 # Step 2

    Prob_Array = []
    for Fitness in FSE:                             # Step 3
        if Flag:                                #IF MultiModal
            prob = MultiModal_P_i(Fitness, FSE_min, FSE_max, Eta)
        else:                                   #IF SingleModal
            prob = SingleModal_P_i(Fitness, FSE_max)
        Prob_Array.append(prob)

    for i in range(len(S)):                             # Step 4
        if rand.random() <= Prob_Array[i]:
            for _ in range(N):
                New_Val = [
                    rand.gauss(S[i][j], Delta) for j in range(len(S[i]))
                ]
                if sum(New_Val) > sum(S[i]):
                    S[i] = New_Val
                    FSE[i] = sum(S[i])

    return S, FSE

def Algorithm_6(NP, G, Delta, Eta, F, Xi, Iteration_Count, DE_Mutator=0.5):
    LAMC_Sol = []
    i = 0
    while i < NP:
        Sol = [rand.random() for _ in range(10)]
        Fitness = F(Sol[0])
        LAMC_Sol.append((Sol, Fitness))
        i += 1

    FS_Max = max(Fitness for _, Fitness in LAMC_Sol)
    FS_Min = min(Fitness for _, Fitness in LAMC_Sol)

    NS = rand.choice(G)

    Niches = Algorithm_1([Sol for Sol, _ in LAMC_Sol], NS)

    Processed_Niches = []
    Niche_Idx = 0
    while Niche_Idx < len(Niches):
        LNiche = Niches[Niche_Idx]
        FS_Max_I = max(F(Sol[0]) for Sol in LNiche)
        FS_Min_I = min(F(Sol[0]) for Sol in LNiche)
        Processed_Niches.append({
            'FS_max_i': FS_Max_I,
            'FS_min_i': FS_Min_I,
            'solutions': LNiche
        })

        Niche_Idx += 1

    Fitted_Solutions = Algorithm_4( NS, Processed_Niches, FS_Max, FS_Min, Eta, F, Xi, DE_Mutator )

    i = 0
    while i < len(Fitted_Solutions): 
        Updated_Sol, Fitness = Fitted_Solutions[i]
        Nearest_Solution, Closest_Fitting = min( LAMC_Sol, key=lambda X: sum(abs(A - B) for A, B in zip(X[0], Updated_Sol))
        )
        if Fitness > Closest_Fitting:
            LAMC_Sol.remove((Nearest_Solution, Closest_Fitting))
            LAMC_Sol.append((Updated_Sol, Fitness))
        i += 1

    LAMC_Sol_Sol, LAMC_Sol_Fitness = zip(*LAMC_Sol)
    LAMC_Sol_Sol, LAMC_Sol_Fitness = Algorithm_5(
        list(LAMC_Sol_Sol), list(LAMC_Sol_Fitness), Delta, 1, Eta
    )

    Iteration_Count -= 1
    if Iteration_Count > 0:
        return Algorithm_6( NP, G, Delta, Eta, F, Xi, Iteration_Count, DE_Mutator )
    else:
        return LAMC_Sol


def Algorithm_7(NP, G, Delta, Eta, F, Xi, Max_Generations):
    LAMS = []
    i = 0
    while i < NP:  
        Sol = [rand.random() for _ in range(10)]
        Fitness = F(Sol[0])  # Using Function_One for Fitness calculation
        LAMS.append((Sol, Fitness))
        i += 1

    FS_Max = max(Fitness for _, Fitness in LAMS)
    FS_Min = min(Fitness for _, Fitness in LAMS)

    Clus_Size = len(LAMS) // 5 if len(LAMS) >= 5 else len(LAMS)
    NS = rand.choice(G)

    Species = Algorithm_2([Sol for Sol, _ in LAMS], [Fit for _, Fit in LAMS], Clus_Size)

    Fitted_Solutions = Algorithm_4(NS, Species, FS_Max, FS_Min, Eta, F, Xi)

    LNiche_Index = 0
    while LNiche_Index < len(Species):
        LNiche = Species[LNiche_Index]
        LAMS_Sol = LNiche["solutions"]
        
        j = 0
        while j < len(Fitted_Solutions):
            Updated_Sol, Fitness = Fitted_Solutions[j]
            Nearest_Solution, Closest_Fitting = min([(Sol, sum(Sol)) for Sol in LAMS_Sol], key=lambda X: sum([abs(A - B) for A, B in zip(X[0], Updated_Sol)]))
            if Fitness > Closest_Fitting:
                LAMS_Sol.remove(Nearest_Solution)
                LAMS_Sol.append(Updated_Sol)
            j += 1
        
        LNiche_Index += 1

    LAMS_Sol, LAMS_Fitness = zip(*LAMS)
    LAMS_Sol, LAMS_Fitness = Algorithm_5(list(LAMS_Sol), list(LAMS_Fitness), Delta, 1, Eta)

    return list(zip(LAMS_Sol, LAMS_Fitness))


def Function_One(x):
    if 0 <=x and x<2.5:
        return 80 * (2.5 - x)
    elif 2.5 <=x and x<5.0:
        return 64 * (x - 2.5)
    elif 5.0 <=x and  x<7.5:
        return 64 * (7.5 - x)
    elif 7.5 <=x and x<12.5:
        return 28 * (x - 7.5)
    elif 12.5 <=x and x<17.5:
        return 28 * (17.5 - x)
    elif 17.5 <=x and x< 22.5:
        return 32 * (x - 17.5)
    elif 22.5 <=x and x <27.5:
        return 32 * (27.5 - x)
    elif 27.5 <=x and x<=30:
        return 80 * (x - 27.5)
    else:
        return 0

def Function_Two(x):
    Fun=np.sin(5*np.pi*x)**6
    return Fun

def Function_Three(x):
    Fun=np.exp(-2*np.log(2)*((x-0.08)/(0.854))**2)* (np.sin(5*np.pi*(x**(3/4)-0.05))**6)
    return Fun

def Function_Four(x,y):
    Fun=(x**2 + y - 11)**2 + (x + y**2 - 7)**2
    return Fun

def Function_Five(x, y):
    Fun=(4-2.1*x**2+(x**4)/3)*x**2+x*y+(4*y**2-4)*y**2
    return Fun

def Function_Six(x):
    size=len(x)
    sum=0
    for i in range(size):
        prod=1
        for j in range(1,6):
            prod=prod*(j*np.cos((j+1)*x[i]+j))
        sum=sum+prod
    return -sum

def Function_Seven(x):
    size = len(x)
    sum=0
    for i in range (size):
        sum+=np.sin(10*np.log(x[i]))
                    
    result =sum/size
    return result

def Function_Eight(x,k):
    size=len(x)
    sum=0
    for i in range(size):
        sum=sum+ (10+9*np.cos(2*np.pi*k[i]*x[i]))
    
    funVal=sum*-1
    return funVal
    

NP = 20
G = [2, 5, 10]
Delta = 0.1
Eta = 1e-4
Xi = 0.5
Iterations = 50

def Fitness_Fx(x):
    return Function_One(x)

archive = Algorithm_6(NP, G, Delta, Eta, Fitness_Fx, Xi, Iterations)

print("Final archive solutions and Fitness:")
for Sol, Fitness in archive:
    print(f"solutions: {Sol}, {Fitness}")



NP = 20
G = [2, 5, 10]
Delta = 0.1
Eta = 1e-4
Xi = 0.5
Iterations = 50

def Fitness_Fx(x):
    return Function_Two(x)

archive = Algorithm_6(NP, G, Delta, Eta, Fitness_Fx, Xi, Iterations)

print("Final archive solutions and Fitness:")
for Sol, Fitness in archive:
    print(f": {Sol}, {Fitness}")
