import sys
import re
import math
#import time


def sigmoid(score):
        overflow = 20.0
        if score > overflow:
                score = overflow
        elif score < -overflow:
                score = -overflow
        exp = math.exp(score)
        return exp / (1 + exp)


def tokenizeDoc(cur_doc):
        return re.findall("\\w+",cur_doc)


def classify(vocabSize,learningRate,regCoefficient,maxIter,size,fileName,weights):

        labels_all = ["Person", "Place", "Work","Species","other"]


        for i in range(0, maxIter):

                A = [[0.0 for x in range(vocabSize)] for y in range(5)]
                k = 0

                curLearn = learningRate/(i+1)**2


                for a in range(0,size):
                        val = [0,0,0,0,0]
                        p = [0.0,0.0,0.0,0.0,0.0]
                        k = k + 1
                        line = sys.stdin.readline()
                        line = line.strip()
                        text = re.split(r'\t+', line)
                        labels = re.split(r'\,+', text[1])
                        curLabels = [labels_all.index(label) for label in labels]
                        features = tokenizeDoc(text[2])


                        for feature_index in range(len(features)):

                                cur_word = hash(features[feature_index])%(vocabSize)

                                if(cur_word < 0):

                                                cur_word += vocabSize

                                features[feature_index] = cur_word


                        for r in range(len(labels_all)):

                                for cur_word in features:


                                        weights[r][cur_word] = weights[r][cur_word]*((1.0 - 2.0*curLearn*regCoefficient)**(k - A[r][cur_word]))
                                        A[r][cur_word] = k
                                        val[r] += weights[r][cur_word]

                                y = r in curLabels

                                p[r] = sigmoid(val[r])

                                for w in features:

                                        weights[r][w] = weights[r][w] + curLearn*(y - p[r])





                for cur_l in range(len(labels_all)):

                        for cur_w in range(vocabSize):

                                weights[cur_l][cur_w] = weights[cur_l][cur_w]*((1.0 - 2.0*curLearn*regCoefficient)**(k - A[cur_l][cur_w]))
                                A[cur_l][cur_w] = k

        for cur_l in range(len(labels_all)):

                        for cur_w in range(vocabSize):

                                weights[cur_l][cur_w] = weights[cur_l][cur_w]*((1.0 - 2.0*curLearn*regCoefficient)**(k - A[cur_l][cur_w]))


        #count = 0
        #total = 0
        with open(fileName) as fp:

                for line in fp:

                        line = line.strip()
                        text = re.split(r'\t+', line)
                        labels = re.split(r'\,+', text[1])
                        features = tokenizeDoc(text[2])


                        prediction = ""

                        for i in range(len(labels_all)):
                                #total += 1

                                label = labels_all[i]

                                prob = 0

                                for word in features:

                                        word_index = hash(word)%vocabSize
                                        if(word_index < 0):
                                                word_index += vocabSize
                                        prob += weights[i][word_index]

                                final_prob = sigmoid(prob)
                                #if(final_prob >= 0.5 and label in labels):
                                        #count+=1
                               # elif(final_prob < 0.5 and label not in labels):
                                        #count += 1


                                if(i != 4):

                                        prediction += label + '\t' + str(final_prob) + ","

                                else:
                                        prediction += label + '\t' + str(final_prob)

                        print(prediction)

                fp.close()
                #print( count/ (total*1.0))

                #print(weights)





if __name__ == '__main__':

        #start = time.time()
        vocabSize = int(sys.argv[1])
        learningRate = float(sys.argv[2])
        regCoefficient = float(sys.argv[3])
        maxIter = int(sys.argv[4])
        size = int(sys.argv[5])
        fileName = sys.argv[6]
        weights = [[0.0 for x in range(vocabSize)] for y in range(5)]
        classify(vocabSize, learningRate, regCoefficient, maxIter, size, fileName, weights)
        #end = time.time()
        #print(end - start)
