from sklearn.preprocessing import OneHotEncoder



def OHotEncoder(X, li):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(X)
    enc.categories_
    li2 = list()
    for i in range(len(li)):
        li2.append([li[i]])
    result = enc.transform(li2).toarray()

    lista1 = list()
    lista2 = list()
    for j in range(len(result)):
        for k in range(len(result[j])):
            lista1.append(result[j][k])
        lista2.append(lista1)
        lista1 = []
    return lista2


def Dose(info):
    name = list()
    lista_only_dose = list()
    lista_only_dose_number  = list()
    lista_number = list()
    for t in range(len(info)):
        name.append(info[t])
        spliit = info[t].split()
        for tt in range(len(spliit)):
            if 'nM' in spliit[tt]:
                lista_only_dose.append(spliit[tt])
    
    for r in range(len(lista_only_dose)):
        for rr in range(len(lista_only_dose[r])):
            if lista_only_dose[r][rr] != "n":
                if lista_only_dose[r][rr] == ",":
                    lista_only_dose_number.append(".")
                else:
                    lista_only_dose_number.append(lista_only_dose[r][rr])
            else:
                break
        lista_number.append(float("".join(lista_only_dose_number))) 
        lista_only_dose_number = []

    return lista_number, name


def LigandOHE(info):
    lista_only_Ligand_st = list()
    for t in range(len(info)):
        spliit = info[t].split()
        if 'hCG' in spliit:
            lista_only_Ligand_st.append('hCG')
        elif 'LH' in spliit:
            lista_only_Ligand_st.append('LH')
        elif 'FSH' in spliit:
            lista_only_Ligand_st.append('FSH')
        else:
            lista_only_Ligand_st.append('None')
    
    X = [['hCG'], ['LH'], ['FSH'], ['None']]
    lista = OHotEncoder(X, lista_only_Ligand_st)
    return lista


def ReceptorOHE(info):
    lista_only_Ligand_st = list()
    for t in range(len(info)):
        spliit = info[t].split()
        if 'LHR' in spliit and 'WT' in spliit:
            lista_only_Ligand_st.append('hLHR') 
        elif 'RLH' in spliit:
            lista_only_Ligand_st.append('hLHR') 
        elif 'hLHR' in spliit:
            lista_only_Ligand_st.append('hLHR') 
        elif 'LHR-T' in spliit:
            lista_only_Ligand_st.append('LHR-T')
        elif 'mLHR' in spliit:
            lista_only_Ligand_st.append('mLHR') 
        elif 'hFSHR' in spliit:
            lista_only_Ligand_st.append('hFSHR') 
        elif 'mFSHR' in spliit:
            lista_only_Ligand_st.append('mFSHR') 
        elif 'FSHR' in spliit:
            lista_only_Ligand_st.append('hFSHR') 
        elif 'LHR' in spliit:
            lista_only_Ligand_st.append('hLHR') 
        else:
            lista_only_Ligand_st.append('None')

    X = [['hLHR'], ['LHR_T'], ['mLHR'], ['hFSHR'], ['mFSHR'], ['None']]
    lista = OHotEncoder(X, lista_only_Ligand_st)
    return lista 


def PerturbationOHE(info):
    lista_only_Ligand_st = list()
    for t in range(len(info)):
        spliit = info[t].split()
        if 'Dyngo4A' in spliit:
            lista_only_Ligand_st.append('Dyngo4A')
        elif 'PitStop' in spliit:
            lista_only_Ligand_st.append('PitStop')
        elif 'Es9-17' in spliit:
            lista_only_Ligand_st.append('Es9_17')
        elif 'YM254890' in spliit:
            lista_only_Ligand_st.append('YM254890')
        elif 'F8' in spliit:
            lista_only_Ligand_st.append('F8')
        elif 'Nb37' in spliit and 'NES' in spliit:
            lista_only_Ligand_st.append('Nb37')
        elif 'Nb37' in spliit and 'CAAX' in spliit:
            lista_only_Ligand_st.append('Nb37')
        elif 'Nb37' in spliit  and '2xFYVE' in spliit:
           lista_only_Ligand_st.append('Nb37')
        elif 'mCherry' in spliit:
            lista_only_Ligand_st.append('mCherry')
        elif 'Control' in spliit:
            lista_only_Ligand_st.append('Control')
        else:
            lista_only_Ligand_st.append('None')

    X = [['Dyngo4A'], ['PitStop'], ['Es9_17'], ['YM254890'], ['F8'], ['Nb37'], ['mCherry'], ['Control'], ['None']]
    lista = OHotEncoder(X, lista_only_Ligand_st)
    return lista