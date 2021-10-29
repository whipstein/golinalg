package eig

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zget24 checks the nonsymmetric eigenvalue (Schur form) problem
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
func zget24(comp bool, jtype int, thresh float64, iseed []int, n int, a, h, ht *mat.CMatrix, w, wt, wtmp *mat.CVector, vs, vs1 *mat.CMatrix, rcdein, rcdvin float64, nslct int, islct []int, isrt int, result *mat.Vector, work *mat.CVector, lwork int, rwork *mat.Vector, bwork *[]bool) (err error) {
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
	if thresh < zero {
		err = fmt.Errorf("thresh < zero: thresh=%v", thresh)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < 1 || a.Rows < n {
		err = fmt.Errorf("a.Rows < 1 || a.Rows < n: a.Rows=%v, n=%v", a.Rows, n)
	} else if vs.Rows < 1 || vs.Rows < n {
		err = fmt.Errorf("vs.Rows < 1 || vs.Rows < n: vs.Rows=%v, n=%v", vs.Rows, n)
	} else if lwork < 2*n {
		err = fmt.Errorf("lwork < 2*n: lwork=%v, n=%v", lwork, n)
	}

	if err != nil {
		gltest.Xerbla2("zget24", err)
		return
	}

	//     Quick return if nothing to do
	for i = 1; i <= 17; i++ {
		result.Set(i-1, -one)
	}

	if n == 0 {
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
		golapack.Zlacpy(Full, n, n, a, h)
		if sdim, rconde, rcondv, iinfo, err = golapack.Zgeesx('V', sort, zslect, 'N', n, h, w, vs, work, lwork, rwork, bwork); err != nil || iinfo != 0 {
			result.Set(1+rsub-1, ulpinv)
			if jtype != 22 {
				fmt.Printf(" zget24: %s returned info=%6d.\n         N=%6d, jtype=%6d, iseed=%5d\n", "Zgeesx1", iinfo, n, jtype, iseed)
			} else {
				fmt.Printf(" zget24: %s returned info=%6d.\n         N=%6d, input example number=%4d\n", "Zgeesx1", iinfo, n, iseed[0])
			}
			err = fmt.Errorf("iinfo=%v", abs(iinfo))
			return
		}
		if isort == 0 {
			goblas.Zcopy(n, w.Off(0, 1), wtmp.Off(0, 1))
		}

		//        Do Test (1) or Test (7)
		result.Set(1+rsub-1, zero)
		for j = 1; j <= n-1; j++ {
			for i = j + 1; i <= n; i++ {
				if h.Get(i-1, j-1) != czero {
					result.Set(1+rsub-1, ulpinv)
				}
			}
		}

		//        Test (2) or (8): Compute norm(A - Q*H*Q') / (norm(A) * N * ULP)
		//
		//        Copy A to VS1, used as workspace
		golapack.Zlacpy(Full, n, n, a, vs1)

		//        Compute Q*H and store in HT.
		if err = goblas.Zgemm(NoTrans, NoTrans, n, n, n, cone, vs, h, czero, ht); err != nil {
			panic(err)
		}

		//        Compute A - Q*H*Q'
		if err = goblas.Zgemm(NoTrans, ConjTrans, n, n, n, -cone, ht, vs, cone, vs1); err != nil {
			panic(err)
		}

		anorm = math.Max(golapack.Zlange('1', n, n, a, rwork), smlnum)
		wnorm = golapack.Zlange('1', n, n, vs1, rwork)

		if anorm > wnorm {
			result.Set(2+rsub-1, (wnorm/anorm)/(float64(n)*ulp))
		} else {
			if anorm < one {
				result.Set(2+rsub-1, (math.Min(wnorm, float64(n)*anorm)/anorm)/(float64(n)*ulp))
			} else {
				result.Set(2+rsub-1, math.Min(wnorm/anorm, float64(n))/(float64(n)*ulp))
			}
		}

		//        Test (3) or (9):  Compute norm( I - Q'*Q ) / ( N * ULP )
		result.Set(3+rsub-1, zunt01('C', n, n, vs, work, lwork, rwork))

		//        Do Test (4) or Test (10)
		result.Set(4+rsub-1, zero)
		for i = 1; i <= n; i++ {
			if h.Get(i-1, i-1) != w.Get(i-1) {
				result.Set(4+rsub-1, ulpinv)
			}
		}

		//        Do Test (5) or Test (11)
		golapack.Zlacpy(Full, n, n, a, ht)
		if sdim, rconde, rcondv, iinfo, err = golapack.Zgeesx('N', sort, zslect, 'N', n, ht, wt, vs, work, lwork, rwork, bwork); err != nil || iinfo != 0 {
			result.Set(5+rsub-1, ulpinv)
			if jtype != 22 {
				fmt.Printf(" zget24: %s returned info=%6d.\n         N=%6d, jtype=%6d, iseed=%5d\n", "Zgeesx2", iinfo, n, jtype, iseed)
			} else {
				fmt.Printf(" zget24: %s returned info=%6d.\n         N=%6d, input example number=%4d\n", "Zgeesx2", iinfo, n, iseed[0])
			}
			err = fmt.Errorf("iinfo=%v", abs(iinfo))
			goto label220
		}

		result.Set(5+rsub-1, zero)
		for j = 1; j <= n; j++ {
			for i = 1; i <= n; i++ {
				if h.Get(i-1, j-1) != ht.Get(i-1, j-1) {
					result.Set(5+rsub-1, ulpinv)
				}
			}
		}

		//        Do Test (6) or Test (12)
		result.Set(6+rsub-1, zero)
		for i = 1; i <= n; i++ {
			if w.Get(i-1) != wt.Get(i-1) {
				result.Set(6+rsub-1, ulpinv)
			}
		}

		//        Do Test (13)
		if isort == 1 {
			result.Set(12, zero)
			knteig = 0
			for i = 1; i <= n; i++ {
				if zslect(w.Get(i - 1)) {
					knteig = knteig + 1
				}
				if i < n {
					if zslect(w.Get(i)) && (!zslect(w.Get(i - 1))) {
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
	if lwork >= (n*(n+1))/2 {
		//        Compute both RCONDE and RCONDV with VS
		sort = 'S'
		result.Set(13, zero)
		result.Set(14, zero)
		golapack.Zlacpy(Full, n, n, a, ht)
		if sdim1, rconde, rcondv, iinfo, err = golapack.Zgeesx('V', sort, zslect, 'B', n, ht, wt, vs1, work, lwork, rwork, bwork); err != nil || iinfo != 0 {
			result.Set(13, ulpinv)
			result.Set(14, ulpinv)
			if jtype != 22 {
				fmt.Printf(" zget24: %s returned info=%6d.\n         N=%6d, jtype=%6d, iseed=%5d\n", "Zgeesx3", iinfo, n, jtype, iseed)
			} else {
				fmt.Printf(" zget24: %s returned info=%6d.\n         N=%6d, input example number=%4d\n", "Zgeesx3", iinfo, n, iseed[0])
			}
			err = fmt.Errorf("iinfo=%v", abs(iinfo))
			goto label220
		}

		//        Perform tests (10), (11), (12), and (13)
		for i = 1; i <= n; i++ {
			if w.Get(i-1) != wt.Get(i-1) {
				result.Set(9, ulpinv)
			}
			for j = 1; j <= n; j++ {
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
		golapack.Zlacpy(Full, n, n, a, ht)
		if sdim1, rcnde1, rcndv1, iinfo, err = golapack.Zgeesx('N', sort, zslect, 'B', n, ht, wt, vs1, work, lwork, rwork, bwork); err != nil || iinfo != 0 {
			result.Set(13, ulpinv)
			result.Set(14, ulpinv)
			if jtype != 22 {
				fmt.Printf(" zget24: %s returned info=%6d.\n         N=%6d, jtype=%6d, iseed=%5d\n", "Zgeesx4", iinfo, n, jtype, iseed)
			} else {
				fmt.Printf(" zget24: %s returned info=%6d.\n         N=%6d, input example number=%4d\n", "Zgeesx4", iinfo, n, iseed[0])
			}
			err = fmt.Errorf("iinfo=%v", abs(iinfo))
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
		for i = 1; i <= n; i++ {
			if w.Get(i-1) != wt.Get(i-1) {
				result.Set(9, ulpinv)
			}
			for j = 1; j <= n; j++ {
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
		golapack.Zlacpy(Full, n, n, a, ht)
		if sdim1, rcnde1, rcndv1, iinfo, err = golapack.Zgeesx('V', sort, zslect, 'E', n, ht, wt, vs1, work, lwork, rwork, bwork); err != nil || iinfo != 0 {
			result.Set(13, ulpinv)
			if jtype != 22 {
				fmt.Printf(" zget24: %s returned info=%6d.\n         N=%6d, jtype=%6d, iseed=%5d\n", "Zgeesx5", iinfo, n, jtype, iseed)
			} else {
				fmt.Printf(" zget24: %s returned info=%6d.\n         N=%6d, input example number=%4d\n", "Zgeesx5", iinfo, n, iseed[0])
			}
			err = fmt.Errorf("iinfo=%v", abs(iinfo))
			goto label220
		}

		//        Perform test (14)
		if rcnde1 != rconde {
			result.Set(13, ulpinv)
		}

		//        Perform tests (10), (11), (12), and (13)
		for i = 1; i <= n; i++ {
			if w.Get(i-1) != wt.Get(i-1) {
				result.Set(9, ulpinv)
			}
			for j = 1; j <= n; j++ {
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
		golapack.Zlacpy(Full, n, n, a, ht)
		if sdim1, rcnde1, rcndv1, iinfo, err = golapack.Zgeesx('N', sort, zslect, 'E', n, ht, wt, vs1, work, lwork, rwork, bwork); err != nil || iinfo != 0 {
			result.Set(13, ulpinv)
			if jtype != 22 {
				fmt.Printf(" zget24: %s returned info=%6d.\n         N=%6d, jtype=%6d, iseed=%5d\n", "Zgeesx6", iinfo, n, jtype, iseed)
			} else {
				fmt.Printf(" zget24: %s returned info=%6d.\n         N=%6d, input example number=%4d\n", "Zgeesx6", iinfo, n, iseed[0])
			}
			err = fmt.Errorf("iinfo=%v", abs(iinfo))
			goto label220
		}

		//        Perform test (14)
		if rcnde1 != rconde {
			result.Set(13, ulpinv)
		}

		//        Perform tests (10), (11), (12), and (13)
		for i = 1; i <= n; i++ {
			if w.Get(i-1) != wt.Get(i-1) {
				result.Set(9, ulpinv)
			}
			for j = 1; j <= n; j++ {
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
		golapack.Zlacpy(Full, n, n, a, ht)
		if sdim1, rcnde1, rcndv1, iinfo, err = golapack.Zgeesx('V', sort, zslect, 'V', n, ht, wt, vs1, work, lwork, rwork, bwork); err != nil || iinfo != 0 {
			result.Set(14, ulpinv)
			if jtype != 22 {
				fmt.Printf(" zget24: %s returned info=%6d.\n         N=%6d, jtype=%6d, iseed=%5d\n", "Zgeesx7", iinfo, n, jtype, iseed)
			} else {
				fmt.Printf(" zget24: %s returned info=%6d.\n         N=%6d, input example number=%4d\n", "Zgeesx7", iinfo, n, iseed[0])
			}
			err = fmt.Errorf("iinfo=%v", abs(iinfo))
			goto label220
		}

		//        Perform test (15)
		if rcndv1 != rcondv {
			result.Set(14, ulpinv)
		}

		//        Perform tests (10), (11), (12), and (13)
		for i = 1; i <= n; i++ {
			if w.Get(i-1) != wt.Get(i-1) {
				result.Set(9, ulpinv)
			}
			for j = 1; j <= n; j++ {
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
		golapack.Zlacpy(Full, n, n, a, ht)
		if sdim1, rcnde1, rcndv1, iinfo, err = golapack.Zgeesx('N', sort, zslect, 'V', n, ht, wt, vs1, work, lwork, rwork, bwork); err != nil || iinfo != 0 {
			result.Set(14, ulpinv)
			if jtype != 22 {
				fmt.Printf(" zget24: %s returned info=%6d.\n         N=%6d, jtype=%6d, iseed=%5d\n", "Zgeesx8", iinfo, n, jtype, iseed)
			} else {
				fmt.Printf(" zget24: %s returned info=%6d.\n         N=%6d, input example number=%4d\n", "Zgeesx8", iinfo, n, iseed[0])
			}
			err = fmt.Errorf("iinfo=%v", abs(iinfo))
			goto label220
		}

		//        Perform test (15)
		if rcndv1 != rcondv {
			result.Set(14, ulpinv)
		}

		//        Perform tests (10), (11), (12), and (13)
		for i = 1; i <= n; i++ {
			if w.Get(i-1) != wt.Get(i-1) {
				result.Set(9, ulpinv)
			}
			for j = 1; j <= n; j++ {
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
		(*seldim) = n
		(*selopt) = 1
		eps = math.Max(ulp, epsin)
		for i = 1; i <= n; i++ {
			ipnt[i-1] = i
			(*selval)[i-1] = false
			selwr.Set(i-1, wtmp.GetRe(i-1))
			selwi.Set(i-1, wtmp.GetIm(i-1))
		}
		for i = 1; i <= n-1; i++ {
			kmin = i
			if isrt == 0 {
				vrimin = wtmp.GetRe(i - 1)
			} else {
				vrimin = wtmp.GetIm(i - 1)
			}
			for j = i + 1; j <= n; j++ {
				if isrt == 0 {
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
		for i = 1; i <= nslct; i++ {
			(*selval)[ipnt[islct[i-1]-1]-1] = true
		}

		//        Compute condition numbers
		golapack.Zlacpy(Full, n, n, a, ht)
		if sdim1, rconde, rcondv, iinfo, err = golapack.Zgeesx('N', 'S', zslect, 'B', n, ht, wt, vs1, work, lwork, rwork, bwork); err != nil || iinfo != 0 {
			result.Set(15, ulpinv)
			result.Set(16, ulpinv)
			fmt.Printf(" zget24: %s returned info=%6d.\n         N=%6d, input example number=%4d\n", "Zgeesx9", iinfo, n, iseed[0])
			err = fmt.Errorf("iinfo=%v", abs(iinfo))
			goto label270
		}

		//        Compare condition number for average of selected eigenvalues
		//        taking its condition number into account
		anorm = golapack.Zlange('1', n, n, a, rwork)
		v = math.Max(float64(n)*eps*anorm, smlnum)
		if anorm == zero {
			v = one
		}
		if v > rcondv {
			tol = one
		} else {
			tol = v / rcondv
		}
		if v > rcdvin {
			tolin = one
		} else {
			tolin = v / rcdvin
		}
		tol = math.Max(tol, smlnum/eps)
		tolin = math.Max(tolin, smlnum/eps)
		if eps*(rcdein-tolin) > rconde+tol {
			result.Set(15, ulpinv)
		} else if rcdein-tolin > rconde+tol {
			result.Set(15, (rcdein-tolin)/(rconde+tol))
		} else if rcdein+tolin < eps*(rconde-tol) {
			result.Set(15, ulpinv)
		} else if rcdein+tolin < rconde-tol {
			result.Set(15, (rconde-tol)/(rcdein+tolin))
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
		if v > rcdvin*rcdein {
			tolin = rcdvin
		} else {
			tolin = v / rcdein
		}
		tol = math.Max(tol, smlnum/eps)
		tolin = math.Max(tolin, smlnum/eps)
		if eps*(rcdvin-tolin) > rcondv+tol {
			result.Set(16, ulpinv)
		} else if rcdvin-tolin > rcondv+tol {
			result.Set(16, (rcdvin-tolin)/(rcondv+tol))
		} else if rcdvin+tolin < eps*(rcondv-tol) {
			result.Set(16, ulpinv)
		} else if rcdvin+tolin < rcondv-tol {
			result.Set(16, (rcondv-tol)/(rcdvin+tolin))
		} else {
			result.Set(16, one)
		}

	label270:
	}

	return
}
