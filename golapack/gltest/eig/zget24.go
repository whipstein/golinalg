package eig

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zget24 checks the nonsymmetric eigenvalue (Schur form) problem
//    expert driver ZGEESX.
//
//    If COMP = .FALSE., the first 13 of the following tests will be
//    be performed on the input matrix A, and also tests 14 and 15
//    if LWORK is sufficiently large.
//    If COMP = .TRUE., all 17 test will be performed.
//
//    (1)     0 if T is in Schur form, 1/ulp otherwise
//           (no sorting of eigenvalues)
//
//    (2)     | A - VS T VS' | / ( n |A| ulp )
//
//      Here VS is the matrix of Schur eigenvectors, and T is in Schur
//      form  (no sorting of eigenvalues).
//
//    (3)     | I - VS VS' | / ( n ulp ) (no sorting of eigenvalues).
//
//    (4)     0     if W are eigenvalues of T
//            1/ulp otherwise
//            (no sorting of eigenvalues)
//
//    (5)     0     if T(with VS) = T(without VS),
//            1/ulp otherwise
//            (no sorting of eigenvalues)
//
//    (6)     0     if eigenvalues(with VS) = eigenvalues(without VS),
//            1/ulp otherwise
//            (no sorting of eigenvalues)
//
//    (7)     0 if T is in Schur form, 1/ulp otherwise
//            (with sorting of eigenvalues)
//
//    (8)     | A - VS T VS' | / ( n |A| ulp )
//
//      Here VS is the matrix of Schur eigenvectors, and T is in Schur
//      form  (with sorting of eigenvalues).
//
//    (9)     | I - VS VS' | / ( n ulp ) (with sorting of eigenvalues).
//
//    (10)    0     if W are eigenvalues of T
//            1/ulp otherwise
//            If workspace sufficient, also compare W with and
//            without reciprocal condition numbers
//            (with sorting of eigenvalues)
//
//    (11)    0     if T(with VS) = T(without VS),
//            1/ulp otherwise
//            If workspace sufficient, also compare T with and without
//            reciprocal condition numbers
//            (with sorting of eigenvalues)
//
//    (12)    0     if eigenvalues(with VS) = eigenvalues(without VS),
//            1/ulp otherwise
//            If workspace sufficient, also compare VS with and without
//            reciprocal condition numbers
//            (with sorting of eigenvalues)
//
//    (13)    if sorting worked and SDIM is the number of
//            eigenvalues which were SELECTed
//            If workspace sufficient, also compare SDIM with and
//            without reciprocal condition numbers
//
//    (14)    if RCONDE the same no matter if VS and/or RCONDV computed
//
//    (15)    if RCONDV the same no matter if VS and/or RCONDE computed
//
//    (16)  |RCONDE - RCDEIN| / cond(RCONDE)
//
//       RCONDE is the reciprocal average eigenvalue condition number
//       computed by ZGEESX and RCDEIN (the precomputed true value)
//       is supplied as input.  cond(RCONDE) is the condition number
//       of RCONDE, and takes errors in computing RCONDE into account,
//       so that the resulting quantity should be O(ULP). cond(RCONDE)
//       is essentially given by norm(A)/RCONDV.
//
//    (17)  |RCONDV - RCDVIN| / cond(RCONDV)
//
//       RCONDV is the reciprocal right invariant subspace condition
//       number computed by ZGEESX and RCDVIN (the precomputed true
//       value) is supplied as input. cond(RCONDV) is the condition
//       number of RCONDV, and takes errors in computing RCONDV into
//       account, so that the resulting quantity should be O(ULP).
//       cond(RCONDV) is essentially given by norm(A)/RCONDE.
func Zget24(comp bool, jtype *int, thresh *float64, iseed *[]int, nounit, n *int, a *mat.CMatrix, lda *int, h, ht *mat.CMatrix, w, wt, wtmp *mat.CVector, vs *mat.CMatrix, ldvs *int, vs1 *mat.CMatrix, rcdein, rcdvin *float64, nslct *int, islct *[]int, isrt *int, result *mat.Vector, work *mat.CVector, lwork *int, rwork *mat.Vector, bwork *[]bool, info *int) {
	var sort byte
	var cone, ctmp, czero complex128
	var anorm, eps, epsin, one, rcnde1, rcndv1, rconde, rcondv, smlnum, tol, tolin, ulp, ulpinv, v, vricmp, vrimin, wnorm, zero float64
	var i, iinfo, isort, itmp, j, kmin, knteig, rsub, sdim, sdim1 int
	ipnt := make([]int, 20)

	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	zero = 0.0
	one = 1.0
	epsin = 5.9605e-8
	selopt := &gltest.Common.Sslct.Selopt
	seldim := &gltest.Common.Sslct.Seldim
	selval := &gltest.Common.Sslct.Selval
	selwr := gltest.Common.Sslct.Selwr
	selwi := gltest.Common.Sslct.Selwi

	//     Check for errors
	(*info) = 0
	if (*thresh) < zero {
		(*info) = -3
	} else if (*nounit) <= 0 {
		(*info) = -5
	} else if (*n) < 0 {
		(*info) = -6
	} else if (*lda) < 1 || (*lda) < (*n) {
		(*info) = -8
	} else if (*ldvs) < 1 || (*ldvs) < (*n) {
		(*info) = -15
	} else if (*lwork) < 2*(*n) {
		(*info) = -24
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGET24"), -(*info))
		return
	}

	//     Quick return if nothing to do
	for i = 1; i <= 17; i++ {
		result.Set(i-1, -one)
	}

	if (*n) == 0 {
		return
	}

	//     Important constants
	smlnum = golapack.Dlamch(SafeMinimum)
	ulp = golapack.Dlamch(Precision)
	ulpinv = one / ulp

	//     Perform tests (1)-(13)
	(*selopt) = 0
	for isort = 0; isort <= 1; isort++ {
		if isort == 0 {
			sort = 'N'
			rsub = 0
		} else {
			sort = 'S'
			rsub = 6
		}

		//        Compute Schur form and Schur vectors, and test them
		golapack.Zlacpy('F', n, n, a, lda, h, lda)
		golapack.Zgeesx('V', sort, Zslect, 'N', n, h, lda, &sdim, w, vs, ldvs, &rconde, &rcondv, work, lwork, rwork, bwork, &iinfo)
		if iinfo != 0 {
			result.Set(1+rsub-1, ulpinv)
			if (*jtype) != 22 {
				fmt.Printf(" ZGET24: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZGEESX1", iinfo, *n, *jtype, *iseed)
			} else {
				fmt.Printf(" ZGET24: %s returned INFO=%6d.\n         N=%6d, INPUT EXAMPLE NUMBER = %4d\n", "ZGEESX1", iinfo, *n, (*iseed)[0])
			}
			(*info) = absint(iinfo)
			return
		}
		if isort == 0 {
			goblas.Zcopy(n, w, func() *int { y := 1; return &y }(), wtmp, func() *int { y := 1; return &y }())
		}

		//        Do Test (1) or Test (7)
		result.Set(1+rsub-1, zero)
		for j = 1; j <= (*n)-1; j++ {
			for i = j + 1; i <= (*n); i++ {
				if h.Get(i-1, j-1) != czero {
					result.Set(1+rsub-1, ulpinv)
				}
			}
		}

		//        Test (2) or (8): Compute norm(A - Q*H*Q') / (norm(A) * N * ULP)
		//
		//        Copy A to VS1, used as workspace
		golapack.Zlacpy(' ', n, n, a, lda, vs1, ldvs)

		//        Compute Q*H and store in HT.
		goblas.Zgemm(NoTrans, NoTrans, n, n, n, &cone, vs, ldvs, h, lda, &czero, ht, lda)

		//        Compute A - Q*H*Q'
		goblas.Zgemm(NoTrans, ConjTrans, n, n, n, toPtrc128(-cone), ht, lda, vs, ldvs, &cone, vs1, ldvs)

		anorm = maxf64(golapack.Zlange('1', n, n, a, lda, rwork), smlnum)
		wnorm = golapack.Zlange('1', n, n, vs1, ldvs, rwork)

		if anorm > wnorm {
			result.Set(2+rsub-1, (wnorm/anorm)/(float64(*n)*ulp))
		} else {
			if anorm < one {
				result.Set(2+rsub-1, (minf64(wnorm, float64(*n)*anorm)/anorm)/(float64(*n)*ulp))
			} else {
				result.Set(2+rsub-1, minf64(wnorm/anorm, float64(*n))/(float64(*n)*ulp))
			}
		}

		//        Test (3) or (9):  Compute norm( I - Q'*Q ) / ( N * ULP )
		Zunt01('C', n, n, vs, ldvs, work, lwork, rwork, result.GetPtr(3+rsub-1))

		//        Do Test (4) or Test (10)
		result.Set(4+rsub-1, zero)
		for i = 1; i <= (*n); i++ {
			if h.Get(i-1, i-1) != w.Get(i-1) {
				result.Set(4+rsub-1, ulpinv)
			}
		}

		//        Do Test (5) or Test (11)
		golapack.Zlacpy('F', n, n, a, lda, ht, lda)
		golapack.Zgeesx('N', sort, Zslect, 'N', n, ht, lda, &sdim, wt, vs, ldvs, &rconde, &rcondv, work, lwork, rwork, bwork, &iinfo)
		if iinfo != 0 {
			result.Set(5+rsub-1, ulpinv)
			if (*jtype) != 22 {
				fmt.Printf(" ZGET24: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZGEESX2", iinfo, *n, *jtype, *iseed)
			} else {
				fmt.Printf(" ZGET24: %s returned INFO=%6d.\n         N=%6d, INPUT EXAMPLE NUMBER = %4d\n", "ZGEESX2", iinfo, *n, (*iseed)[0])
			}
			(*info) = absint(iinfo)
			goto label220
		}

		result.Set(5+rsub-1, zero)
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*n); i++ {
				if h.Get(i-1, j-1) != ht.Get(i-1, j-1) {
					result.Set(5+rsub-1, ulpinv)
				}
			}
		}

		//        Do Test (6) or Test (12)
		result.Set(6+rsub-1, zero)
		for i = 1; i <= (*n); i++ {
			if w.Get(i-1) != wt.Get(i-1) {
				result.Set(6+rsub-1, ulpinv)
			}
		}

		//        Do Test (13)
		if isort == 1 {
			result.Set(12, zero)
			knteig = 0
			for i = 1; i <= (*n); i++ {
				if Zslect(w.Get(i - 1)) {
					knteig = knteig + 1
				}
				if i < (*n) {
					if Zslect(w.Get(i+1-1)) && (!Zslect(w.Get(i - 1))) {
						result.Set(12, ulpinv)
					}
				}
			}
			if sdim != knteig {
				result.Set(12, ulpinv)
			}
		}

	}

	//     If there is enough workspace, perform tests (14) and (15)
	//     as well as (10) through (13)
	if (*lwork) >= ((*n)*((*n)+1))/2 {
		//        Compute both RCONDE and RCONDV with VS
		sort = 'S'
		result.Set(13, zero)
		result.Set(14, zero)
		golapack.Zlacpy('F', n, n, a, lda, ht, lda)
		golapack.Zgeesx('V', sort, Zslect, 'B', n, ht, lda, &sdim1, wt, vs1, ldvs, &rconde, &rcondv, work, lwork, rwork, bwork, &iinfo)
		if iinfo != 0 {
			result.Set(13, ulpinv)
			result.Set(14, ulpinv)
			if (*jtype) != 22 {
				fmt.Printf(" ZGET24: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZGEESX3", iinfo, *n, *jtype, *iseed)
			} else {
				fmt.Printf(" ZGET24: %s returned INFO=%6d.\n         N=%6d, INPUT EXAMPLE NUMBER = %4d\n", "ZGEESX3", iinfo, *n, (*iseed)[0])
			}
			(*info) = absint(iinfo)
			goto label220
		}

		//        Perform tests (10), (11), (12), and (13)
		for i = 1; i <= (*n); i++ {
			if w.Get(i-1) != wt.Get(i-1) {
				result.Set(9, ulpinv)
			}
			for j = 1; j <= (*n); j++ {
				if h.Get(i-1, j-1) != ht.Get(i-1, j-1) {
					result.Set(10, ulpinv)
				}
				if vs.Get(i-1, j-1) != vs1.Get(i-1, j-1) {
					result.Set(11, ulpinv)
				}
			}
		}
		if sdim != sdim1 {
			result.Set(12, ulpinv)
		}

		//        Compute both RCONDE and RCONDV without VS, and compare
		golapack.Zlacpy('F', n, n, a, lda, ht, lda)
		golapack.Zgeesx('N', sort, Zslect, 'B', n, ht, lda, &sdim1, wt, vs1, ldvs, &rcnde1, &rcndv1, work, lwork, rwork, bwork, &iinfo)
		if iinfo != 0 {
			result.Set(13, ulpinv)
			result.Set(14, ulpinv)
			if (*jtype) != 22 {
				fmt.Printf(" ZGET24: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZGEESX4", iinfo, *n, *jtype, *iseed)
			} else {
				fmt.Printf(" ZGET24: %s returned INFO=%6d.\n         N=%6d, INPUT EXAMPLE NUMBER = %4d\n", "ZGEESX4", iinfo, *n, (*iseed)[0])
			}
			(*info) = absint(iinfo)
			goto label220
		}

		//        Perform tests (14) and (15)
		if rcnde1 != rconde {
			result.Set(13, ulpinv)
		}
		if rcndv1 != rcondv {
			result.Set(14, ulpinv)
		}

		//        Perform tests (10), (11), (12), and (13)
		for i = 1; i <= (*n); i++ {
			if w.Get(i-1) != wt.Get(i-1) {
				result.Set(9, ulpinv)
			}
			for j = 1; j <= (*n); j++ {
				if h.Get(i-1, j-1) != ht.Get(i-1, j-1) {
					result.Set(10, ulpinv)
				}
				if vs.Get(i-1, j-1) != vs1.Get(i-1, j-1) {
					result.Set(11, ulpinv)
				}
			}
		}
		if sdim != sdim1 {
			result.Set(12, ulpinv)
		}

		//        Compute RCONDE with VS, and compare
		golapack.Zlacpy('F', n, n, a, lda, ht, lda)
		golapack.Zgeesx('V', sort, Zslect, 'E', n, ht, lda, &sdim1, wt, vs1, ldvs, &rcnde1, &rcndv1, work, lwork, rwork, bwork, &iinfo)
		if iinfo != 0 {
			result.Set(13, ulpinv)
			if (*jtype) != 22 {
				fmt.Printf(" ZGET24: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZGEESX5", iinfo, *n, *jtype, *iseed)
			} else {
				fmt.Printf(" ZGET24: %s returned INFO=%6d.\n         N=%6d, INPUT EXAMPLE NUMBER = %4d\n", "ZGEESX5", iinfo, *n, (*iseed)[0])
			}
			(*info) = absint(iinfo)
			goto label220
		}

		//        Perform test (14)
		if rcnde1 != rconde {
			result.Set(13, ulpinv)
		}

		//        Perform tests (10), (11), (12), and (13)
		for i = 1; i <= (*n); i++ {
			if w.Get(i-1) != wt.Get(i-1) {
				result.Set(9, ulpinv)
			}
			for j = 1; j <= (*n); j++ {
				if h.Get(i-1, j-1) != ht.Get(i-1, j-1) {
					result.Set(10, ulpinv)
				}
				if vs.Get(i-1, j-1) != vs1.Get(i-1, j-1) {
					result.Set(11, ulpinv)
				}
			}
		}
		if sdim != sdim1 {
			result.Set(12, ulpinv)
		}

		//        Compute RCONDE without VS, and compare
		golapack.Zlacpy('F', n, n, a, lda, ht, lda)
		golapack.Zgeesx('N', sort, Zslect, 'E', n, ht, lda, &sdim1, wt, vs1, ldvs, &rcnde1, &rcndv1, work, lwork, rwork, bwork, &iinfo)
		if iinfo != 0 {
			result.Set(13, ulpinv)
			if (*jtype) != 22 {
				fmt.Printf(" ZGET24: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZGEESX6", iinfo, *n, *jtype, *iseed)
			} else {
				fmt.Printf(" ZGET24: %s returned INFO=%6d.\n         N=%6d, INPUT EXAMPLE NUMBER = %4d\n", "ZGEESX6", iinfo, *n, (*iseed)[0])
			}
			(*info) = absint(iinfo)
			goto label220
		}

		//        Perform test (14)
		if rcnde1 != rconde {
			result.Set(13, ulpinv)
		}

		//        Perform tests (10), (11), (12), and (13)
		for i = 1; i <= (*n); i++ {
			if w.Get(i-1) != wt.Get(i-1) {
				result.Set(9, ulpinv)
			}
			for j = 1; j <= (*n); j++ {
				if h.Get(i-1, j-1) != ht.Get(i-1, j-1) {
					result.Set(10, ulpinv)
				}
				if vs.Get(i-1, j-1) != vs1.Get(i-1, j-1) {
					result.Set(11, ulpinv)
				}
			}
		}
		if sdim != sdim1 {
			result.Set(12, ulpinv)
		}

		//        Compute RCONDV with VS, and compare
		golapack.Zlacpy('F', n, n, a, lda, ht, lda)
		golapack.Zgeesx('V', sort, Zslect, 'V', n, ht, lda, &sdim1, wt, vs1, ldvs, &rcnde1, &rcndv1, work, lwork, rwork, bwork, &iinfo)
		if iinfo != 0 {
			result.Set(14, ulpinv)
			if (*jtype) != 22 {
				fmt.Printf(" ZGET24: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZGEESX7", iinfo, *n, *jtype, *iseed)
			} else {
				fmt.Printf(" ZGET24: %s returned INFO=%6d.\n         N=%6d, INPUT EXAMPLE NUMBER = %4d\n", "ZGEESX7", iinfo, *n, (*iseed)[0])
			}
			(*info) = absint(iinfo)
			goto label220
		}

		//        Perform test (15)
		if rcndv1 != rcondv {
			result.Set(14, ulpinv)
		}

		//        Perform tests (10), (11), (12), and (13)
		for i = 1; i <= (*n); i++ {
			if w.Get(i-1) != wt.Get(i-1) {
				result.Set(9, ulpinv)
			}
			for j = 1; j <= (*n); j++ {
				if h.Get(i-1, j-1) != ht.Get(i-1, j-1) {
					result.Set(10, ulpinv)
				}
				if vs.Get(i-1, j-1) != vs1.Get(i-1, j-1) {
					result.Set(11, ulpinv)
				}
			}
		}
		if sdim != sdim1 {
			result.Set(12, ulpinv)
		}

		//        Compute RCONDV without VS, and compare
		golapack.Zlacpy('F', n, n, a, lda, ht, lda)
		golapack.Zgeesx('N', sort, Zslect, 'V', n, ht, lda, &sdim1, wt, vs1, ldvs, &rcnde1, &rcndv1, work, lwork, rwork, bwork, &iinfo)
		if iinfo != 0 {
			result.Set(14, ulpinv)
			if (*jtype) != 22 {
				fmt.Printf(" ZGET24: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZGEESX8", iinfo, *n, *jtype, *iseed)
			} else {
				fmt.Printf(" ZGET24: %s returned INFO=%6d.\n         N=%6d, INPUT EXAMPLE NUMBER = %4d\n", "ZGEESX8", iinfo, *n, (*iseed)[0])
			}
			(*info) = absint(iinfo)
			goto label220
		}

		//        Perform test (15)
		if rcndv1 != rcondv {
			result.Set(14, ulpinv)
		}

		//        Perform tests (10), (11), (12), and (13)
		for i = 1; i <= (*n); i++ {
			if w.Get(i-1) != wt.Get(i-1) {
				result.Set(9, ulpinv)
			}
			for j = 1; j <= (*n); j++ {
				if h.Get(i-1, j-1) != ht.Get(i-1, j-1) {
					result.Set(10, ulpinv)
				}
				if vs.Get(i-1, j-1) != vs1.Get(i-1, j-1) {
					result.Set(11, ulpinv)
				}
			}
		}
		if sdim != sdim1 {
			result.Set(12, ulpinv)
		}

	}

label220:
	;

	//     If there are precomputed reciprocal condition numbers, compare
	//     computed values with them.
	if comp {
		//        First set up SELOPT, SELDIM, SELVAL, SELWR and SELWI so that
		//        the logical function Zslect selects the eigenvalues specified
		//        by NSLCT, ISLCT and ISRT.
		(*seldim) = (*n)
		(*selopt) = 1
		eps = maxf64(ulp, epsin)
		for i = 1; i <= (*n); i++ {
			ipnt[i-1] = i
			(*selval)[i-1] = false
			selwr.Set(i-1, wtmp.GetRe(i-1))
			selwi.Set(i-1, wtmp.GetIm(i-1))
		}
		for i = 1; i <= (*n)-1; i++ {
			kmin = i
			if (*isrt) == 0 {
				vrimin = wtmp.GetRe(i - 1)
			} else {
				vrimin = wtmp.GetIm(i - 1)
			}
			for j = i + 1; j <= (*n); j++ {
				if (*isrt) == 0 {
					vricmp = wtmp.GetRe(j - 1)
				} else {
					vricmp = wtmp.GetIm(j - 1)
				}
				if vricmp < vrimin {
					kmin = j
					vrimin = vricmp
				}
			}
			ctmp = wtmp.Get(kmin - 1)
			wtmp.Set(kmin-1, wtmp.Get(i-1))
			wtmp.Set(i-1, ctmp)
			itmp = ipnt[i-1]
			ipnt[i-1] = ipnt[kmin-1]
			ipnt[kmin-1] = itmp
		}
		for i = 1; i <= (*nslct); i++ {
			(*selval)[ipnt[(*islct)[i-1]-1]-1] = true
		}

		//        Compute condition numbers
		golapack.Zlacpy('F', n, n, a, lda, ht, lda)
		golapack.Zgeesx('N', 'S', Zslect, 'B', n, ht, lda, &sdim1, wt, vs1, ldvs, &rconde, &rcondv, work, lwork, rwork, bwork, &iinfo)
		if iinfo != 0 {
			result.Set(15, ulpinv)
			result.Set(16, ulpinv)
			fmt.Printf(" ZGET24: %s returned INFO=%6d.\n         N=%6d, INPUT EXAMPLE NUMBER = %4d\n", "ZGEESX9", iinfo, *n, (*iseed)[0])
			(*info) = absint(iinfo)
			goto label270
		}

		//        Compare condition number for average of selected eigenvalues
		//        taking its condition number into account
		anorm = golapack.Zlange('1', n, n, a, lda, rwork)
		v = maxf64(float64(*n)*eps*anorm, smlnum)
		if anorm == zero {
			v = one
		}
		if v > rcondv {
			tol = one
		} else {
			tol = v / rcondv
		}
		if v > (*rcdvin) {
			tolin = one
		} else {
			tolin = v / (*rcdvin)
		}
		tol = maxf64(tol, smlnum/eps)
		tolin = maxf64(tolin, smlnum/eps)
		if eps*((*rcdein)-tolin) > rconde+tol {
			result.Set(15, ulpinv)
		} else if (*rcdein)-tolin > rconde+tol {
			result.Set(15, ((*rcdein)-tolin)/(rconde+tol))
		} else if (*rcdein)+tolin < eps*(rconde-tol) {
			result.Set(15, ulpinv)
		} else if (*rcdein)+tolin < rconde-tol {
			result.Set(15, (rconde-tol)/((*rcdein)+tolin))
		} else {
			result.Set(15, one)
		}

		//        Compare condition numbers for right invariant subspace
		//        taking its condition number into account
		if v > rcondv*rconde {
			tol = rcondv
		} else {
			tol = v / rconde
		}
		if v > (*rcdvin)*(*rcdein) {
			tolin = (*rcdvin)
		} else {
			tolin = v / (*rcdein)
		}
		tol = maxf64(tol, smlnum/eps)
		tolin = maxf64(tolin, smlnum/eps)
		if eps*((*rcdvin)-tolin) > rcondv+tol {
			result.Set(16, ulpinv)
		} else if (*rcdvin)-tolin > rcondv+tol {
			result.Set(16, ((*rcdvin)-tolin)/(rcondv+tol))
		} else if (*rcdvin)+tolin < eps*(rcondv-tol) {
			result.Set(16, ulpinv)
		} else if (*rcdvin)+tolin < rcondv-tol {
			result.Set(16, (rcondv-tol)/((*rcdvin)+tolin))
		} else {
			result.Set(16, one)
		}

	label270:
	}
}
