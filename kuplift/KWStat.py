import math
import numpy as np
from scipy.special import gammaln, betainc, gammainc, gammaincc

class KWStat:

    # Tables de cache pour les calculs
    dvLnFactorial = []
    dvLnBell = []
    dvLnStar = []
    dvC0Max = []

    # Constantes pour la taille des tables
    nLnFactorialTableSize = 128000
    nLnBellTableMaxN = 100
    nLnStarTableMaxN = 2000

    @staticmethod
    def require(condition):
        if not condition:
            raise ValueError("Precondition failed")

    # ============================
    # Fonctions sur vecteurs
    # ============================

    @staticmethod
    def Min(dvValues):
        """ Retourne la valeur minimale d'un vecteur """
        KWStat.require(dvValues is not None)
        return min(dvValues)

    @staticmethod
    def Max(dvValues):
        """ Retourne la valeur maximale d'un vecteur """
        KWStat.require(dvValues is not None)
        return max(dvValues)

    @staticmethod
    def Mean(dvValues):
        """ Retourne la moyenne arithmétique d'un vecteur """
        KWStat.require(dvValues is not None)
        if len(dvValues) > 0:
            return sum(dvValues) / len(dvValues)
        return 0

    @staticmethod
    def GeometricMean(dvValues):
        """ Moyenne géométrique (valeurs positives) """
        KWStat.require(dvValues is not None)
        dLogSum = 0
        for v in dvValues:
            assert v >= 0
            if v == 0:
                return 0
            dLogSum += math.log(v)
        if len(dvValues) > 0:
            return math.exp(dLogSum / len(dvValues))
        return 0

    @staticmethod
    def StandardDeviation(dvValues):
        """ Écart-type d'un vecteur """
        KWStat.require(dvValues is not None)
        n = len(dvValues)
        if n == 0:
            return 0
        dSum = sum(dvValues)
        dSquareSum = sum([v * v for v in dvValues])
        variance = (dSquareSum - (dSum * dSum) / n) / n
        return math.sqrt(abs(variance))

    @staticmethod
    def TValue(dvValues1, dvValues2):
        """ T-value pour la comparaison de deux vecteurs de même taille """
        KWStat.require(dvValues1 is not None and dvValues2 is not None)
        KWStat.require(len(dvValues1) == len(dvValues2))
        n = len(dvValues1)
        if n == 0:
            return 0
        diff_sum = 0
        diff_sq_sum = 0
        for i in range(n):
            d = dvValues1[i] - dvValues2[i]
            diff_sum += d
            diff_sq_sum += d * d
        dMean = diff_sum / n
        dStd = math.sqrt(abs((diff_sq_sum - diff_sum * diff_sum / n) / n))
        return dMean * math.sqrt(n) / (dStd + 1e-5)

    # ============================
    # Fonctions sur la loi normale
    # ============================

    @staticmethod
    def Normal(dX, dMean, dStandardDeviation):
        """ Loi normale N(dMean, dStandardDeviation) """
        assert dStandardDeviation > 0
        return KWStat.StandardNormal((dX - dMean) / dStandardDeviation)

    @staticmethod
    def StandardNormal(dX):
        """ Loi normale standard N(0,1) """
        return 1 - 0.5 * KWStat.Erfc(dX / math.sqrt(2))

    @staticmethod
    def InvNormal(dProb, dMean, dStandardDeviation):
        """ Inverse loi normale (approximation par dichotomie) """
        dTolerance = 1e-7
        dMax = 1e20
        dLowerX = -dMax
        dUpperX = dMax
        KWStat.require(0 <= dProb <= 1)
        KWStat.require(dStandardDeviation > 0)

        while (dUpperX - dLowerX) > dTolerance * (abs(dLowerX) + abs(dUpperX)):
            dNewX = (dLowerX + dUpperX) / 2
            dNewXVal = KWStat.StandardNormal(dNewX)
            if dNewXVal < dProb:
                dLowerX = dNewX
            else:
                dUpperX = dNewX
        return dLowerX

    @staticmethod
    def InvStandardNormal(dProb):
        """ Inverse loi normale standard N(0,1) """
        dTolerance = 1e-7
        dMax = 1e20
        dLowerX = -dMax
        dUpperX = dMax
        KWStat.require(0 <= dProb <= 1)

        while (dUpperX - dLowerX) > dTolerance * (abs(dLowerX) + abs(dUpperX)):
            dNewX = (dLowerX + dUpperX) / 2
            dNewXVal = KWStat.StandardNormal(dNewX)
            if dNewXVal < dProb:
                dLowerX = dNewX
            else:
                dUpperX = dNewX
        return dLowerX

    # ============================
    # Fonction de probabilité binomiale
    # ============================

    @staticmethod
    def BinomialProb(n, dProb, k):
        """ Probabilité binomiale P(X=k) """
        KWStat.require(0 <= n)
        KWStat.require(0 <= k <= n)
        KWStat.require(0 <= dProb <= 1)
        return math.comb(n, k) * (dProb ** k) * ((1 - dProb) ** (n - k))

    # ============================
    # Fonctions d'erreur
    # ============================

    @staticmethod
    def Erf(dX):
        """ Fonction erreur erf(x) """
        return math.erf(dX)

    @staticmethod
    def Erfc(dX):
        """ Fonction erreur complémentaire erfc(x) """
        return math.erfc(dX)

    # ============================
    # Loi de Student
    # ============================

    @staticmethod
    def Student(dTValue, ndf):
        """ Loi de Student en fonction de t et degrés de liberté """
        KWStat.require(dTValue >= 0)
        KWStat.require(ndf >= 1)
        return betainc(ndf / 2.0, 0.5, ndf / (ndf + dTValue * dTValue))

    @staticmethod
    def InvStudent(dProb, ndf):
        """ Inverse loi de Student (approximation par dichotomie) """
        dTolerance = 1e-7
        dMax = 1e20
        dLowerX = 0
        dUpperX = dMax
        KWStat.require(0 <= dProb <= 1)
        KWStat.require(ndf >= 1)

        # Cas particulier
        if dProb == 0:
            return 0
        if dProb == 1:
            return dMax

        while (dUpperX - dLowerX) > dTolerance * (abs(dLowerX) + abs(dUpperX)):
            dNewX = (dLowerX + dUpperX) / 2
            dNewXVal = KWStat.Student(dNewX, ndf)
            if dNewXVal > dProb:
                dLowerX = dNewX
            else:
                dUpperX = dNewX
        return dLowerX

    # ============================
    # Logarithme de factorielle avec table cache
    # ============================

    @staticmethod
    def LnFactorial(nValue):
        """ Logarithme de la factorielle n! avec cache """
        KWStat.require(nValue >= 0)
        if len(KWStat.dvLnFactorial) == 0:
            # Initialisation du tableau
            KWStat.dvLnFactorial = [0.0] * KWStat.nLnFactorialTableSize
            for i in range(1, KWStat.nLnFactorialTableSize):
                KWStat.dvLnFactorial[i] = KWStat.dvLnFactorial[i - 1] + math.log(i)
        if nValue < len(KWStat.dvLnFactorial):
            return KWStat.dvLnFactorial[nValue]
        else:
            return KWStat.LnGamma(nValue + 1)

    # ============================
    # Nombre de Bell généralisé
    # ============================

    @staticmethod
    def LnBell(n, k):
        """ Logarithme du nombre de Bell généralisé B(n,k) """
        KWStat.require(n >= 1)
        KWStat.require(1 <= k <= n)
        if len(KWStat.dvLnBell) == 0:
            KWStat.ComputeLnBellTable()
        if n < KWStat.nLnBellTableMaxN:
            return KWStat.dvLnBell[(n - 1) * KWStat.nLnBellTableMaxN + (k - 1)]
        else:
            return KWStat.ComputeLnBellValue(n, k)

    # ============================
    # Codage universel des entiers naturels (Rissanen)
    # ============================

    @staticmethod
    def NaturalNumbersUniversalCodeLength(n):
        """ Code universel pour n >= 1 (en bits) """
        dC0 = 2.86511
        dLog2 = math.log(2)
        return (math.log(dC0) / dLog2) + KWStat.LnStar(n)

    @staticmethod
    def BoundedNaturalNumbersUniversalCodeLength(n, nMax):
        """ Code universel pour 1 <= n <= nMax """
        dLog2 = math.log(2)
        return (math.log(KWStat.C0Max(nMax)) / dLog2) + KWStat.LnStar(n)

    # ============================
    # Log* (log star) fonction
    # ============================

    @staticmethod
    def LnStar(n):
        """ Log* (log star) de n """
        KWStat.require(n > 0)
        if len(KWStat.dvLnStar) == 0:
            KWStat.ComputeLnStarAndC0MaxTables()
        if n - 1 < len(KWStat.dvLnStar):
            return KWStat.dvLnStar[n - 1]
        else:
            dCost = 0
            dLogI = math.log(n) / math.log(2)
            while dLogI > 0:
                dCost += dLogI
                dLogI = math.log(dLogI) / math.log(2)
            return dCost

    @staticmethod
    def C0Max(nMax):
        """ C0Max(nMax) - approximation ou table """
        # Pour simplification, on retourne 1 ou on peut faire une approximation
        return 1

    # ============================
    # Calcul des tables
    # ============================

    @staticmethod
    def ComputeLnStarAndC0MaxTables():
        """ Calcule et remplit les tables dvLnStar et dvC0Max """
        dLog2 = math.log(2)
        size = KWStat.nLnStarTableMaxN
        KWStat.dvLnStar = [0.0] * size
        KWStat.dvC0Max = [0.0] * size
        KWStat.dvLnStar[0] = 0.0
        KWStat.dvC0Max[0] = 1.0
        for i in range(1, size):
            # log*(i+1) base 2
            dCost = 0
            dLogI = math.log(i + 1) / dLog2
            while dLogI > 0:
                dCost += dLogI
                dLogI = math.log(dLogI) / dLog2
            KWStat.dvLnStar[i] = dCost
            KWStat.dvC0Max[i] = KWStat.dvC0Max[i - 1] + 2 ** (-dCost)

    # ============================
    # Calcul de lnBell(n,k)
    # ============================

    @staticmethod
    def ComputeLnBellTable():
        """ Calcule la table des nombres de Bell généralisés log(B(n,k)) """
        size = KWStat.nLnBellTableMaxN * KWStat.nLnBellTableMaxN
        KWStat.dvLnBell = [0.0] * size
        # Calcul des nombres de Stirling de seconde espèce
        stirling = [0.0] * size
        for i in range(KWStat.nLnBellTableMaxN):
            stirling[i * KWStat.nLnBellTableMaxN + 0] = 1
            if i > 0:
                stirling[i * KWStat.nLnBellTableMaxN + 1] = 2 ** i - 1
            if i > 1:
                stirling[i * KWStat.nLnBellTableMaxN + i] = 1
            if i > 2:
                stirling[i * KWStat.nLnBellTableMaxN + i - 1] = i * (i + 1) / 2

        # Recurrence pour autres valeurs
        for i in range(1, KWStat.nLnBellTableMaxN):
            for j in range(2, i):
                stirling[i * KWStat.nLnBellTableMaxN + j] = (
                    stirling[(i - 1) * KWStat.nLnBellTableMaxN + j - 1]
                    + (j + 1) * stirling[(i - 1) * KWStat.nLnBellTableMaxN + j]
                )

        # Calcul des nombres de Bell par sommation
        for i in range(1, KWStat.nLnBellTableMaxN + 1):
            dBell = 0
            for j in range(1, i + 1):
                dBell += stirling[(i - 1) * KWStat.nLnBellTableMaxN + j - 1]
                KWStat.dvLnBell[(i - 1) * KWStat.nLnBellTableMaxN + j - 1] = math.log(dBell)

    @staticmethod
    def ComputeLnBellValue(n, k):
        """ Calcul du log(Bell(n,k)) via série """
        dEpsilon = 1e-6 / k
        nSerieSize = 20
        # Série pour exp(-1)
        if not hasattr(KWStat, 'dvInvExpSerie'):
            KWStat.dvInvExpSerie = [0.0] * nSerieSize
            dInvExp = math.exp(-1)
            dFactor = 1
            dTerm = 1
            KWStat.dvInvExpSerie[0] = 1
            for i in range(1, nSerieSize):
                dFactor *= -i
                dTerm += 1 / dFactor
                KWStat.dvInvExpSerie[i] = dTerm
            assert abs(KWStat.dvInvExpSerie[-1] - dInvExp) < 1e-12

        def get_inv_exp(k_minus_i):
            if k_minus_i >= nSerieSize:
                return math.exp(-1)
            return KWStat.dvInvExpSerie[k_minus_i]

        # Recherche i0
        i0 = KWStat.ComputeMainBellTermIndex(n, k)
        dLnTermI0 = n * math.log(i0) - KWStat.LnFactorial(i0)

        # Série autour de i0
        dBellSerie = 0
        for i in range(i0, k + 1):
            dInvExpFactor = get_inv_exp(k - i)
            dTerm = math.exp(n * math.log(i) - KWStat.LnFactorial(i) - dLnTermI0)
            dBellSerie += dTerm * dInvExpFactor
            if dTerm < dEpsilon:
                break

        for i in range(i0 - 1, 0, -1):
            dInvExpFactor = get_inv_exp(k - i)
            dTerm = math.exp(n * math.log(i) + KWStat.LnFactorial(i0) - KWStat.LnFactorial(i))
            dBellSerie += dTerm * dInvExpFactor
            if dTerm < dEpsilon:
                break

        return math.log(dBellSerie) + dLnTermI0

    @staticmethod
    def ComputeMainBellTermIndex(n, k):
        """ Recherche dichotomique de i maximisant i^n / i! """
        iMin = int(math.floor(n / math.log(n))) if n > 1 else 1
        iMax = int(math.ceil(n / (math.log(n) - math.log(math.log(n))))) if n > 1 else 2
        while iMax - iMin > 1:
            iMid = (iMin + iMax) // 2
            val = iMid * math.log(iMid)
            if val > n:
                iMax = iMid
            else:
                iMin = iMid
        return min(iMin, k)

    # ============================
    # Probabilités Chi2
    # ============================

    @staticmethod
    def LnProb(dChi2, ndf):
        """ Logarithme de la loi du Chi2 """
        KWStat.require(dChi2 >= 0)
        KWStat.require(ndf >= 1)
        if ndf == 1:
            return math.log(0.5 * math.erfc(math.sqrt(dChi2 / 2)))
        elif ndf == 2:
            return -dChi2 / 2
        else:
            return gammaln(ndf / 2.0) + (ndf / 2.0 - 1) * math.log(dChi2 / 2) - dChi2 / 2

    @staticmethod
    def ProbLevel(dChi2, ndf):
        """ Niveau de la probabilité du Chi2 en -log10 """
        return -KWStat.LnProb(dChi2, ndf) / math.log(10)

    @staticmethod
    def Chi2(dChi2, ndf):
        """ Probabilité du Chi2 """
        return math.exp(KWStat.LnProb(dChi2, ndf))

    @staticmethod
    def InvChi2(dProb, ndf):
        """ Inverse de la loi Chi2 par dichotomie """
        dTolerance = 1e-7
        dMax = 1e20
        dLowerX = 0
        dUpperX = dMax
        KWStat.require(0 <= dProb <= 1)
        KWStat.require(ndf >= 1)

        if dProb == 0:
            return 0
        if dProb == 1:
            return dMax

        dLnProb = math.log(dProb)
        while (dUpperX - dLowerX) > dTolerance * (abs(dLowerX) + abs(dUpperX)):
            dNewX = (dLowerX + dUpperX) / 2
            dNewXVal = KWStat.LnProb(dNewX, ndf)
            if dNewXVal > dLnProb:
                dLowerX = dNewX
            else:
                dUpperX = dNewX
        return dLowerX

    # ============================
    # Fonction de test
    # ============================

    @staticmethod
    def Test():
        """ Exemple de tests pour vérifier le fonctionnement """
        print("Test de KWStat:")

        # Test Min, Max, Mean
        vect = [1, 2, 3, 4, 5]
        print("Min:", KWStat.Min(vect))
        print("Max:", KWStat.Max(vect))
        print("Mean:", KWStat.Mean(vect))
        print("GeometricMean:", KWStat.GeometricMean(vect))
        print("StandardDeviation:", KWStat.StandardDeviation(vect))
        print("TValue:", KWStat.TValue([1,2,3], [1,2,2]))

        # Test loi normale
        print("Normal(0, 0, 1):", KWStat.Normal(0, 0, 1))
        print("StandardNormal(1):", KWStat.StandardNormal(1))
        print("InvNormal(0.95, 0, 1):", KWStat.InvNormal(0.95, 0, 1))
        print("InvStandardNormal(0.95):", KWStat.InvStandardNormal(0.95))

        # Test loi Chi2
        print("LnProb(10, 5):", KWStat.LnProb(10, 5))
        print("Chi2(10, 5):", KWStat.Chi2(10, 5))
        #print("InvChi2(0.95, 5):", KWStat.InvChi2(0.95, 5))
        print("ProbLevel(10, 5):", KWStat.ProbLevel(10, 5))

        # Test LnFactorial
        print("LnFactorial(10):", KWStat.LnFactorial(10))
        print("LnFactorial(100):", KWStat.LnFactorial(100))

        # Test LnBell
        print("LnBell(5, 3):", KWStat.LnBell(5, 3))

        # Test code universel
        print("Code universel n=10:", KWStat.NaturalNumbersUniversalCodeLength(10))
        print("Code borné n=10, nMax=100:", KWStat.BoundedNaturalNumbersUniversalCodeLength(10, 100))
        print("log*(10):", KWStat.LnStar(10))

    @staticmethod
    def Test2():
        print("=== Test de KWStat ===\n")
        # Si vous souhaitez sauter certains tests, vous pouvez définir une variable
        bSkipIt = False

        # Test de la table de Student
        if not bSkipIt:
            print("Table de Student")
            print("ndf\t0.10\t0.05\t0.01")
            for i in range(1, 11):
                print(f"{i}\t{KWStat.InvStudent(0.10, i):.4f}\t{KWStat.InvStudent(0.05, i):.4f}\t{KWStat.InvStudent(0.01, i):.4f}")
            print()

        # Logarithme de la factorielle
        if not bSkipIt:
            print("Log(Factorielle)")
            dResult = 0
            for j in range(100):
                for i in range(10000):
                    dResult += KWStat.LnFactorial(i)
            print(dResult)
            print()

        # LnBell
        if not bSkipIt:
            print("Ln(Bell)")
            print("i\\k", end="\t")
            for k in range(1, 21):
                print(f"{k}", end="\t")
            print()
            for i in range(1, 21):
                print(f"{i}", end="\t")
                for k in range(1, 21):
                    if i <= k:
                        print(f"{KWStat.LnBell(k, i):.4f}", end="\t")
                    else:
                        print("\t", end="")
                print()
            print()

        # Codage universel de Rissanen
        if not bSkipIt:
            print("Rissanen universal code for integer")
            print("N\tC0Max(N)\tln(C0Max(N))")
            for i in range(1, 10):
                print(f"{i}\t{KWStat.C0Max(i):.4f}\t{math.log(KWStat.C0Max(i)):.4f}")
            i = 10
            for j in range(1, 9):
                print(f"{i}\t{KWStat.C0Max(i):.4f}\t{math.log(KWStat.C0Max(i)):.4f}")
                i *= 10

            # Affichage pour différentes bornes
            ivC0Max = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000]
            print("N\tLn2*(N)", end="\t")
            for c in ivC0Max:
                print(f"UCL(N, {c})", end="\t")
            print("UCL(N)")
            for N in range(1, 101):
                print(f"{N}\t{KWStat.LnStar(N) * math.log(2):.4f}", end="\t")
                for c in ivC0Max:
                    if N <= c:
                        print(f"{KWStat.BoundedNaturalNumbersUniversalCodeLength(N, c):.4f}", end="\t")
                    else:
                        print("\t", end="")
                print(f"{KWStat.NaturalNumbersUniversalCodeLength(N):.4f}")
        print()
        
