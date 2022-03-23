#!/usr/bin/env python
import os.path
import numpy
import sys


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

    if listProblemDescription[0] == '0':
        # TODO: we can't use this method when restrictions are != <=
        startSimplexIterations(
            initialMatrix, numberDesicionVars, headers, rowsDescription, solutionFileName)
    elif listProblemDescription[0] == '1':
        print("Gran M")
        if listProblemDescription[1] == 'max':
            print("max")
        elif listProblemDescription[1] == "min":
            print("min")
        else:
            print("invalid optimization method")
    elif listProblemDescription[0] == '2':
        print("2 fases")
    else:
        print("invalid entered method ")


def startSimplexIterations(matrix, vnBNumber, H, RD, outputLocation):
    # Check if all the row[0][i] with i<VnBNumber  are >= 0
    estado = 0
    writeToFile(f'Estado: {estado}', outputLocation)
    # estado
    # matrix
    str_matrix = getPrintableMatrix(
        H, RD, matrix)
    writeToFile(str_matrix, outputLocation)

    # Start the iterations
    while True:
        estado += 1
        if isRowPositive(matrix[0], len(matrix[0])):
            # then you have the best possible outcome
            # TODO: you can give the final answer in here
            print("FINISHED")
            break
        else:
            #   TODO: check if this is still valid(?)
            #   First get the less row[0][i] with i<VnBNumber - Thats the COLUMNA PIVOTE
            cp_index = getIndexForLessN(matrix[0], len(matrix[0]))
            CP = matrix[:, cp_index]
            LD = matrix[:, len(matrix[0]) - 1]
            #   Then for each item in LD (starting with 1 - ignore the 0) (this is the matrix[0][matrix.len()-1]) do item/Columna Pivote)
            #   Get the index of CP that is lesser of all the devisions (ignore the LD when value is 0)
            #   This index is the FILA PIVOTE
            fp_index = getIndexLesserWhileDivByCP(LD, CP)

            # @TODO: if the fp_index == -1 then you don't have more iterations, the result is not acotado
            if fp_index == -1:
                writeToFile("NO ACOTADO", outputLocation)
                break

            FP = matrix[fp_index]
            #   Interseccion entre COLUMNA PIVAOTE y FILA PIVOTE matrix[CP][FP] makes the numero pivote
            NP = matrix[fp_index][cp_index]
            entrante = H[cp_index + 1]
            saliente = RD[fp_index]
            response = f'VB entrante: {entrante}, VB saliente: {saliente}, Número Pivot: {NP}'

            # variable basica que sale lp_index
            # Now set the all CP to 0 except the fp_index, fp_index = 1
            # then operación de reglón: all the FP need to be / NP
            # CP already has the reference
            for i in range(len(FP)):
                FP[i] = FP[i] / NP
            # not calculate the rest of the rows
            # for all rows
            oldCP = numpy.array(CP, copy=True)
            for i in range(len(matrix)):
                # if not row pivote
                if i != fp_index:
                    currentRow = matrix[i]
                    for j in range(len(matrix[i])):
                        if oldCP[i] > 0:
                            currentRow[j] = currentRow[j] - \
                                (abs(oldCP[i]) * FP[j])
                        else:
                            currentRow[j] = currentRow[j] + \
                                (abs(oldCP[i]) * FP[j])

            # response
            writeToFile(response, outputLocation)
            writeToFile(f'Estado {estado}', outputLocation)
            # TODO: remember to update the RD, with the variable entrante
            str_matrix = getPrintableMatrix(
                H, RD, matrix)
            writeToFile(str_matrix, outputLocation)
            writeToFile(
                f'Respuesta Parcial: {getPartialAnwser(matrix, vnBNumber)}', outputLocation)


def formatFloatToPrint(num):
    return "{:.2f}".format(num)


def getPartialAnwser(matrix, end):
    response = formatFloatToPrint(matrix[0][0])
    # use only the variables de desicion
    for i in range(1, end):
        response += "," + formatFloatToPrint(matrix[0][i])
    # Now the LD col
    LD = matrix[:, len(matrix[0]) - 1]
    for i in range(1, len(LD)):
        response += "," + formatFloatToPrint(LD[i])
    return f'U = {formatFloatToPrint(LD[0])},({response})'


def getIndexLesserWhileDivByCP(ld, cp):
    # TODO: can happen its a SOLUCION DEGENERADA
    resultIndex = -1
    for i in range(1, len(ld)):
        # omit 0 because is undefined
        if cp[i] > 0:
            if (resultIndex == -1) or (ld[i] / cp[i] < ld[resultIndex]/cp[resultIndex]):
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
    print(content)
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
            contentToPrint += formatFloatToPrint(contentItem) + tabChar
        contentToPrint += enterChar
    return contentToPrint


# Run the project
main()
