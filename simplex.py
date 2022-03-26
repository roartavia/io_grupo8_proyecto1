#!/usr/bin/env python
import os.path
import numpy
import sys
import cmath


def main():
    # @TODO: implement the [-h] flag to run the help flow
    if len(sys.argv) <= 1:
        displayHelp()
        return
    if sys.argv[1] == "-h":
        displayHelp()
        return

    # Read the file, is going to be a format like:
    # 0,max,2,3 - (${method type [0=Simplex, 1=GranM, 2=DosFases]}, ${max or min}, ${decision variables}, ${restriccions #})
    # 3,5 - (coeficientes de la función objetivo)
    # 2,1,<=,6 (coeficientes de las restricciones y signo de restricción)
    # -1,3,<=,9 - (coeficientes de las restricciones y signo de restricción)
    # 0,1,<=,4 - (coeficientes de las restricciones y signo de restricción)
    fileName = sys.argv[1]
    solutionFileName = os.path.splitext(fileName)[0] + "_solucion.txt"

    if not os.path.isfile(fileName):
        print("File does not exist.")
        return

    # @TODO: we are expecting the file to have a perfect format (?) - add error handling
    # Start the parsing
    f = open(fileName, "r")
    lines = f.readlines()

    # lines[0] - problem info
    listProblemDescription = lines[0].strip().split(",")
    # lines[1] - coeficientes funcion objetivo
    listCoefficientFnObj = lines[1].strip().split(",")
    # lines[2:] - restricciones
    # For each line remove the new line char '\n'
    # result: ['2', '1', '<=', '6']
    listRestrictions = []
    for item in lines[2:]:
        listRestrictions.append(item.strip().split(","))

    # TODO: this iteration is only for max and with all restricitions <= - we need to check if there is >= and = restrictions

    # This is a possible way to manage the content
    # For the matrix we only need the content, not the headers
    # the header and first col, we are going to use

    # 1. Clean the function objetivo to be U - (variable decision 1) - (variable decision) 2 = 0
    # NOTE: basic variables = amount of restrictions
    # 2. Now we should turn all restrictions to be = (achive this by adding the basic variables)
    # TODO: Step 3 and 2 need to be implemented - for now we expect to have the restrictions in the form <=
    # 3. Create matrix
    # cols = number of restrictions + number of variables + 2 (VB & LD)
    # rows = 2 (VB & U) + len(basic variables) @NOTE: basic variables = # of restrictions for now
    numberDesicionVars = int(listProblemDescription[2])
    numberRestrictions = int(listProblemDescription[3])
    numberOfTotalVars = numberDesicionVars + numberRestrictions
    # cols = (amout of total vars) + LD
    # rows = (amout of restrictions or basic vars) + U
    # 3. Create matrix
    initialMatrix = buildMatrix(
        listCoefficientFnObj, listRestrictions, listProblemDescription)
    # Numpy can' manage the matrix with the headers, headers and row description are manage in 2 separte arrays
    headers = ["VB"]
    for i in range(numberOfTotalVars):
        headers.append("x" + str(i + 1))
    headers.append("LD")
    # when finish add the LD
    rowsDescription = ["U"]
    i = numberDesicionVars + 1
    for j in range(numberRestrictions):
        rowsDescription.append("x" + str(i + j))

    someTestingText = getPrintableMatrix(
        headers, rowsDescription, initialMatrix)
    # print(someTestingText)
    # writeToFile(someTestingText, solutionFileName)

    # startSimplexIterations(                                            #puede borrarse
    # initialMatrix, numberDesicionVars, headers, rowsDescription)

    # ---------------------------------------------------------
    if listProblemDescription[0] == '0':
        # TODO:simplex
        print("simplex")
        startSimplexIterations(initialMatrix, numberDesicionVars, headers, rowsDescription)
    elif listProblemDescription[0] == '1':
        # TODO: Gran M
        #print(listRestrictions)

        vecTemp = []
        numberOfS = 0
        numberOfA = 0
        numberOfS2 = 0  # variable de apoyo

        # cuenta la cantida de variables nuevas
        for i in listRestrictions:
            positionSize = len(i) - 2
            if i[positionSize] == '<=':
                numberOfS += 1
            elif i[positionSize] == '=':
                numberOfA += 1
            elif i[positionSize] == '>=':
                numberOfS +=1
                numberOfA += 1

        #Agrega la cantida de variables nuevas a Z y las restricciones

        for i in range(numberDesicionVars+numberOfS+numberOfA + 1):
            vecTemp.append(complex(0, 0))
        for i in listRestrictions:
            for x in range(numberOfS+numberOfA):
                i.insert(-2, '0')

        # determina el signo de M
        if listProblemDescription[1] == 'max':
            Msing = -1
        elif listProblemDescription[1] == "min":
            Msing = 1
        else:
            print("invalid optimization method")

        # coloca los valores no imaginarios dentro de Z
        for x in range(numberDesicionVars):
            vecTemp[x] = int(listCoefficientFnObj[x]) * -1

        numberOfA=0   # se reutiliza la variable

        # ciclo que asigna los valores de las nuevas variables a las restricciones
        for i in listRestrictions:
            positionSize = len(i) - 2

            if i[positionSize] == '<=':
                i[numberDesicionVars+numberOfS2]='1'
                numberOfS2 += 1

            elif i[positionSize] == '=':
                i[numberDesicionVars + numberOfA+numberOfS] = '1'
                numberOfA += 1

                # agrega los valores imaginarios en Z
                for x in range(numberDesicionVars):
                    vecTemp[x] += complex(0, int(i[x]) * Msing)
                vecTemp[-1] += (complex(0, int(i[-1])))

            elif i[positionSize] == '>=':

                i[numberDesicionVars + numberOfS2] = '1'
                numberOfS2 += 1

                i[numberDesicionVars + numberOfA + numberOfS] = '1'
                numberOfA += 1

                # Agrega los numeros imaginarios y la S
                for x in range(numberDesicionVars):
                    vecTemp[x] += complex(0, int(i[x]) * Msing)
                vecTemp[-1] += (complex(0, int(i[-1])))
                vecTemp.insert(-1, complex(1, 1))
            else:
                print("wrong sign")


        #borra los = de las restricciones
        for i in listRestrictions:
            i.pop(-2)

        #transforma el el tipo de las restricciones
        for i in listRestrictions:
            for x in range(len(i)):
                i[x] = float(i[x])

        print(listRestrictions)
        listCoefficientFnObj = vecTemp
        print(listCoefficientFnObj)

        #****************************** creacion de matriz
        Mheader=["VB"]
        for i in range(numberDesicionVars):
            Mheader.append("X" + str(i + 1))
        for i in range(numberOfS):
            Mheader.append("S" + str(i+1))
        for i in range(numberOfA):
            Mheader.append("A" + str(i+1))
        headers.append("LD")
        print(Mheader)

        MrowsDescription = ["U"]
        for i in range(numberRestrictions):
            MrowsDescription.append("X" + str(i + 1+numberDesicionVars))
        print(MrowsDescription)

        matrixBigM=buildMatrixForBigM(listRestrictions,listCoefficientFnObj)
        #print(matrixBigM)

        startSimplexIterations(matrixBigM, numberDesicionVars, Mheader, MrowsDescription)

        #********************************
    elif listProblemDescription[0] == '2':
        # TODO: 2 fase
        print("2 fases")
    else:
        print("invalid entered method ")

    # -------------------------------------------------------


# VnBNumber - number of variables no basicas


def startSimplexIterations(matrix, vnBNumber, H, RD):
    # Check if all the row[0][i] with i<VnBNumber  are >= 0
    if isRowPositive(matrix[0], vnBNumber):
        # then you have the best possible outcome
        print("FINISHED")
        return
    else:
        #   TODO: check if this is still valid(?)
        #   First get the less row[0][i] with i<VnBNumber - Thats the COLUMNA PIVOTE
        cp_index = getIndexForLessN(matrix[0], vnBNumber)
        CP = matrix[:, cp_index]
        LD = matrix[:, len(matrix[0]) - 1]
        print(matrix)
        print("CP")
        print(cp_index)
        print(CP)
        print("LD")
        print(LD)
        #   Then for each item in LD (starting with 1 - ignore the 0) (this is the matrix[0][matrix.len()-1]) do item/Columna Pivote)
        #   Get the index of CP that is lesser of all the devisions (ignore the LD when value is 0)
        #   This index is the FILA PIVOTE
        fp_index = getIndexLesserWhileDivByCP(LD, CP)
        print(fp_index)
        FP = matrix[fp_index]
        print("FP")
        print(FP)
        #   Interseccion entre COLUMNA PIVAOTE y FILA PIVOTE matrix[CP][FP] makes the numero pivote
        NP = matrix[fp_index][cp_index]
        print("NP")
        print(NP)
        entrante = H[cp_index + 1]
        saliente = RD[fp_index]
        response = f'VB entrante: {entrante}, VB saliente: {saliente}, Número Pivot: {NP}'
        # print estado
        # print estado
        # print respuesta parcial
        print(response)
        # variable basica que sale lp_index
        # Now set the all CP to 0 except the fp_index, fp_index = 1
        # then operación de reglón: all the FP need to be / NP
        # CP already has the reference
        for i in range(len(FP)):
            FP[i] = FP[i] / NP
        print(matrix)
        for i in range(len(CP)):
            if fp_index == i:
                # print a 1
                CP[i] = 1
            else:
                CP[i] = 0
        print(matrix)
        return


#---------------------------------
def buildMatrixForBigM(restrictions, fnsObjetivo):
    cols = len(fnsObjetivo)
    # because the fnObjetivo
    rows = len(restrictions) + 1
    matrix = numpy.zeros([rows, cols]).astype(complex)
    # add fn headers
    for col_index in range(len(fnsObjetivo)):
        matrix[0][col_index] = fnsObjetivo[col_index]
    # add content
    for row_index in range(len(restrictions)):
        restriction = restrictions[row_index]
        for col_index in range(len(restriction)):
            matrix[row_index+1][col_index] = restriction[col_index]
    return matrix
#--------------------------------



def getIndexLesserWhileDivByCP(ld, cp):
    resultIndex = -1
    for i in range(1, len(ld)):
        # omit 0 because is undefined
        if cp[i] != 0:
            if (resultIndex == -1) or (ld[i] / cp[i] < ld[resultIndex] / cp[resultIndex]):
                resultIndex = i
    return resultIndex


def getIndexForLessN(row, end=-1):
    resultIndex = 0
    if end == -1:
        end = len(row)
    for i in range(1, end):
        if row[i] < row[resultIndex]:
            resultIndex = i
    return resultIndex


def isRowPositive(row, end):
    for i in range(0, end):
        item = row[i]
        if item < 0:
            return False
    return True


def displayHelp():
    # TODO: diplay the correct message
    print("here the help")


def writeToFile(content, fileName):
    if os.path.exists(fileName):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    file = open(fileName, append_write)
    file.write(content + '\n')
    file.close()


def buildMatrix(coeficientesFn, coeficientesRestr, problemDescr):
    # cols = number of restrictions + number of variables + 2 (VB & LD)
    # rows = 2 (VB & U) + len(basic variables) @NOTE: basic variables = # of restrictions for now
    numberDesicionVars = int(problemDescr[2])
    numberRestrictions = int(problemDescr[3])
    numberOfTotalVars = numberDesicionVars + numberRestrictions
    # cols = (amout of total vars) + LD
    # rows = (amout of restrictions or basic vars) + U
    matrix = numpy.zeros(
        [(numberRestrictions + 1), (numberOfTotalVars + 1)])
    currentRowIndex = 0
    # We can use the numberDesicionVars
    for coeficiente in coeficientesFn:
        # Move to left side of the expression
        matrix[0][currentRowIndex] = (float(coeficiente)) * -1
        currentRowIndex += 1

    diagonalStartIndex = numberDesicionVars
    for i in range(numberRestrictions):
        for j in range(numberDesicionVars):
            matrix[i + 1][j] = float(coeficientesRestr[i][j])
        # diagonal
        matrix[i + 1][diagonalStartIndex] = float(1)
        diagonalStartIndex += 1

    # Now set the LD
    lastRowIndex = numberOfTotalVars
    for i in range(numberRestrictions):
        matrix[i + 1][lastRowIndex] = coeficientesRestr[i][len(
            coeficientesRestr[i]) - 1]
    return matrix


def getPrintableMatrix(headers, rowsDescr, content):
    contentToPrint = ""
    tabChar = "\t"
    enterChar = "\n"
    for header in headers:
        contentToPrint += header + tabChar
    contentToPrint += enterChar
    for i in range(len(rowsDescr)):
        rowInfo = rowsDescr[i]
        contentToPrint += rowInfo + tabChar
        currentContentRow = content[i]
        for contentItem in currentContentRow:
            contentToPrint += str(contentItem) + tabChar
        contentToPrint += enterChar
    return contentToPrint


# Run the project
main()
