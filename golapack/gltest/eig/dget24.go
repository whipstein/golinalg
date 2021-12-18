package eig

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// dget24 checks the nonsymmetric eigenvalue (Schur form) problem
//    expert driver Dgeesx.
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
//    (4)     0     if WR+math.Sqrt(-1)*WI are eigenvalues of T
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
//    (10)    0     if WR+math.Sqrt(-1)*WI are eigenvalues of T
//            1/ulp otherwise
//            If workspace sufficient, also compare WR, WI with and
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
//       computed by Dgeesx and RCDEIN (the precomputed true value)
//       is supplied as input.  cond(RCONDE) is the condition number
//       of RCONDE, and takes errors in computing RCONDE into account,
//       so that the resulting quantity should be O(ULP). cond(RCONDE)
//       is essentially given by norm(A)/RCONDV.
//
//    (17)  |RCONDV - RCDVIN| / cond(RCONDV)
//
//       RCONDV is the reciprocal right invariant subspace condition
//       number computed by Dgeesx and RCDVIN (the precomputed true
//       value) is supplied as input. cond(RCONDV) is the condition
//       number of RCONDV, and takes errors in computing RCONDV into
//       account, so that the resulting quantity should be O(ULP).
//       cond(RCONDV) is essentially given by norm(A)/RCONDE.
func dget24(comp bool, jtype int, thresh float64, iseed []int, nounit int, n int, a, h, ht *mat.Matrix, wr, wi, wrt, wit, wrtmp, witmp *mat.Vector, vs, vs1 *mat.Matrix, rcdein, rcdvin float64, nslct int, islct *[]int, result, work *mat.Vector, lwork int, iwork *[]int, bwork *[]bool) (info int, err error) {
	var sort byte
	var anorm, eps, epsin, one, rcnde1, rcndv1, rconde, rcondv, smlnum, tmp, tol, tolin, ulp, ulpinv, v, vimin, vrmin, wnorm, zero float64
	var i, iinfo, isort, itmp, j, kmin, knteig, liwork, rsub, sdim, sdim1 int
	ipnt := make([]int, 20)

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
		info = -3
		err = fmt.Errorf("thresh < zero: thresh=%v", thresh)
	} else if nounit <= 0 {
		info = -5
		err = fmt.Errorf("nounit <= 0: nounit=%v", nounit)
	} else if n < 0 {
		info = -6
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < 1 || a.Rows < n {
		info = -8
		err = fmt.Errorf("a.Rows < 1 || a.Rows < n: a.Rows=%v, n=%v", a.Rows, n)
	} else if vs.Rows < 1 || vs.Rows < n {
		info = -18
		err = fmt.Errorf("vs.Rows < 1 || vs.Rows < n: vs.Rows=%v, n=%v", vs.Rows, n)
	} else if lwork < 3*n {
		info = -26
		err = fmt.Errorf("lwork < 3*n: lwork=%v, n=%v", lwork, n)
	}

	if err != nil {
		gltest.Xerbla2("dget24", err)
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
	liwork = n * n
	for isort = 0; isort <= 1; isort++ {
		if isort == 0 {
			sort = 'N'
			rsub = 0
		} else {
			sort = 'S'
			rsub = 6
		}

		//        Compute Schur form and Schur vectors, and test them
		golapack.Dlacpy(Full, n, n, a, h)
		if sdim, rconde, rcondv, iinfo, err = golapack.Dgeesx('V', sort, dslect, 'N', n, h, wr, wi, vs, work, lwork, iwork, liwork, bwork); iinfo != 0 && iinfo != n+2 {
			result.Set(1+rsub-1, ulpinv)
			if jtype != 22 {
				fmt.Printf(" dget24: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dgeesx1", iinfo, n, jtype, iseed)
			} else {
				fmt.Printf(" dget24: %s returned info=%6d.\n         n=%6d, input example number = %4d\n", "Dgeesx1", iinfo, n, iseed[0])
			}
			info = abs(iinfo)
			return
		}
		if isort == 0 {
			wrtmp.Copy(n, wr, 1, 1)
			witmp.Copy(n, wi, 1, 1)
		}

		//        Do Test (1) or Test (7)
		result.Set(1+rsub-1, zero)
		for j = 1; j <= n-2; j++ {
			for i = j + 2; i <= n; i++ {
				if h.Get(i-1, j-1) != zero {
					result.Set(1+rsub-1, ulpinv)
				}
			}
		}
		for i = 1; i <= n-2; i++ {
			if h.Get(i, i-1) != zero && h.Get(i+2-1, i) != zero {
				result.Set(1+rsub-1, ulpinv)
			}
		}
		for i = 1; i <= n-1; i++ {
			if h.Get(i, i-1) != zero {
				if h.Get(i-1, i-1) != h.Get(i, i) || h.Get(i-1, i) == zero || math.Copysign(one, h.Get(i, i-1)) == math.Copysign(one, h.Get(i-1, i)) {
					result.Set(1+rsub-1, ulpinv)
				}
			}
		}

		//        Test (2) or (8): Compute norm(A - Q*H*Q') / (norm(A) * N * ULP)
		//
		//        Copy A to VS1, used as workspace
		golapack.Dlacpy(Full, n, n, a, vs1)

		//        Compute Q*H and store in HT.
		err = ht.Gemm(NoTrans, NoTrans, n, n, n, one, vs, h, zero)

		//        Compute A - Q*H*Q'
		err = vs1.Gemm(NoTrans, Trans, n, n, n, -one, ht, vs, one)

		anorm = math.Max(golapack.Dlange('1', n, n, a, work), smlnum)
		wnorm = golapack.Dlange('1', n, n, vs1, work)

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
		result.Set(3+rsub-1, dort01('C', n, n, vs, work, lwork))

		//        Do Test (4) or Test (10)
		result.Set(4+rsub-1, zero)
		for i = 1; i <= n; i++ {
			if h.Get(i-1, i-1) != wr.Get(i-1) {
				result.Set(4+rsub-1, ulpinv)
			}
		}
		if n > 1 {
			if h.Get(1, 0) == zero && wi.Get(0) != zero {
				result.Set(4+rsub-1, ulpinv)
			}
			if h.Get(n-1, n-1-1) == zero && wi.Get(n-1) != zero {
				result.Set(4+rsub-1, ulpinv)
			}
		}
		for i = 1; i <= n-1; i++ {
			if h.Get(i, i-1) != zero {
				tmp = math.Sqrt(math.Abs(h.Get(i, i-1))) * math.Sqrt(math.Abs(h.Get(i-1, i)))
				result.Set(4+rsub-1, math.Max(result.Get(4+rsub-1), math.Abs(wi.Get(i-1)-tmp)/math.Max(ulp*tmp, smlnum)))
				result.Set(4+rsub-1, math.Max(result.Get(4+rsub-1), math.Abs(wi.Get(i)+tmp)/math.Max(ulp*tmp, smlnum)))
			} else if i > 1 {
				if h.Get(i, i-1) == zero && h.Get(i-1, i-1-1) == zero && wi.Get(i-1) != zero {
					result.Set(4+rsub-1, ulpinv)
				}
			}
		}

		//        Do Test (5) or Test (11)
		golapack.Dlacpy(Full, n, n, a, ht)
		if sdim, rconde, rcondv, iinfo, err = golapack.Dgeesx('N', sort, dslect, 'N', n, ht, wrt, wit, vs, work, lwork, iwork, liwork, bwork); iinfo != 0 && iinfo != n+2 {
			result.Set(5+rsub-1, ulpinv)
			if jtype != 22 {
				fmt.Printf(" dget24: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dgeesx2", iinfo, n, jtype, iseed)
			} else {
				fmt.Printf(" dget24: %s returned info=%6d.\n         n=%6d, input example number = %4d\n", "Dgeesx2", iinfo, n, iseed[0])
			}
			info = abs(iinfo)
			goto label250
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
			if wr.Get(i-1) != wrt.Get(i-1) || wi.Get(i-1) != wit.Get(i-1) {
				result.Set(6+rsub-1, ulpinv)
			}
		}

		//        Do Test (13)
		if isort == 1 {
			result.Set(12, zero)
			knteig = 0
			for i = 1; i <= n; i++ {
				if dslect(wr.GetPtr(i-1), wi.GetPtr(i-1)) || dslect(wr.GetPtr(i-1), toPtrf64(-wi.Get(i-1))) {
					knteig = knteig + 1
				}
				if i < n {
					if (dslect(wr.GetPtr(i), wi.GetPtr(i)) || dslect(wr.GetPtr(i), toPtrf64(-wi.Get(i)))) && (!(dslect(wr.GetPtr(i-1), wi.GetPtr(i-1)) || dslect(wr.GetPtr(i-1), toPtrf64(-wi.Get(i-1))))) && iinfo != n+2 {
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
	if lwork >= n+(n*n)/2 {
		//        Compute both RCONDE and RCONDV with VS
		sort = 'S'
		result.Set(13, zero)
		result.Set(14, zero)
		golapack.Dlacpy(Full, n, n, a, ht)
		if sdim1, rconde, rcondv, iinfo, err = golapack.Dgeesx('V', sort, dslect, 'B', n, ht, wrt, wit, vs1, work, lwork, iwork, liwork, bwork); iinfo != 0 && iinfo != n+2 {
			result.Set(13, ulpinv)
			result.Set(14, ulpinv)
			if jtype != 22 {
				fmt.Printf(" dget24: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dgeesx3", iinfo, n, jtype, iseed)
			} else {
				fmt.Printf(" dget24: %s returned info=%6d.\n         n=%6d, input example number = %4d\n", "Dgeesx3", iinfo, n, iseed[0])
			}
			info = abs(iinfo)
			goto label250
		}

		//        Perform tests (10), (11), (12), and (13)
		for i = 1; i <= n; i++ {
			if wr.Get(i-1) != wrt.Get(i-1) || wi.Get(i-1) != wit.Get(i-1) {
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
		golapack.Dlacpy(Full, n, n, a, ht)
		if sdim1, rcnde1, rcndv1, iinfo, err = golapack.Dgeesx('N', sort, dslect, 'B', n, ht, wrt, wit, vs1, work, lwork, iwork, liwork, bwork); iinfo != 0 && iinfo != n+2 {
			result.Set(13, ulpinv)
			result.Set(14, ulpinv)
			if jtype != 22 {
				fmt.Printf(" dget24: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dgeesx4", iinfo, n, jtype, iseed)
			} else {
				fmt.Printf(" dget24: %s returned info=%6d.\n         n=%6d, input example number = %4d\n", "Dgeesx4", iinfo, n, iseed[0])
			}
			info = abs(iinfo)
			goto label250
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
			if wr.Get(i-1) != wrt.Get(i-1) || wi.Get(i-1) != wit.Get(i-1) {
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
		golapack.Dlacpy(Full, n, n, a, ht)
		if sdim1, rcnde1, rcndv1, iinfo, err = golapack.Dgeesx('V', sort, dslect, 'E', n, ht, wrt, wit, vs1, work, lwork, iwork, liwork, bwork); iinfo != 0 && iinfo != n+2 {
			result.Set(13, ulpinv)
			if jtype != 22 {
				fmt.Printf(" dget24: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dgeesx5", iinfo, n, jtype, iseed)
			} else {
				fmt.Printf(" dget24: %s returned info=%6d.\n         n=%6d, input example number = %4d\n", "Dgeesx5", iinfo, n, iseed[0])
			}
			info = abs(iinfo)
			goto label250
		}

		//        Perform test (14)
		if rcnde1 != rconde {
			result.Set(13, ulpinv)
		}

		//        Perform tests (10), (11), (12), and (13)
		for i = 1; i <= n; i++ {
			if wr.Get(i-1) != wrt.Get(i-1) || wi.Get(i-1) != wit.Get(i-1) {
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
		golapack.Dlacpy(Full, n, n, a, ht)
		if sdim1, rcnde1, rcndv1, iinfo, err = golapack.Dgeesx('N', sort, dslect, 'E', n, ht, wrt, wit, vs1, work, lwork, iwork, liwork, bwork); iinfo != 0 && iinfo != n+2 {
			result.Set(13, ulpinv)
			if jtype != 22 {
				fmt.Printf(" dget24: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dgeesx6", iinfo, n, jtype, iseed)
			} else {
				fmt.Printf(" dget24: %s returned info=%6d.\n         n=%6d, input example number = %4d\n", "Dgeesx6", iinfo, n, iseed[0])
			}
			info = abs(iinfo)
			goto label250
		}

		//        Perform test (14)
		if rcnde1 != rconde {
			result.Set(13, ulpinv)
		}

		//        Perform tests (10), (11), (12), and (13)
		for i = 1; i <= n; i++ {
			if wr.Get(i-1) != wrt.Get(i-1) || wi.Get(i-1) != wit.Get(i-1) {
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
		golapack.Dlacpy(Full, n, n, a, ht)
		if sdim1, rcnde1, rcndv1, iinfo, err = golapack.Dgeesx('V', sort, dslect, 'V', n, ht, wrt, wit, vs1, work, lwork, iwork, liwork, bwork); iinfo != 0 && iinfo != n+2 {
			result.Set(14, ulpinv)
			if jtype != 22 {
				fmt.Printf(" dget24: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dgeesx7", iinfo, n, jtype, iseed)
			} else {
				fmt.Printf(" dget24: %s returned info=%6d.\n         n=%6d, input example number = %4d\n", "Dgeesx7", iinfo, n, iseed[0])
			}
			info = abs(iinfo)
			goto label250
		}

		//        Perform test (15)
		if rcndv1 != rcondv {
			result.Set(14, ulpinv)
		}

		//        Perform tests (10), (11), (12), and (13)
		for i = 1; i <= n; i++ {
			if wr.Get(i-1) != wrt.Get(i-1) || wi.Get(i-1) != wit.Get(i-1) {
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
		golapack.Dlacpy(Full, n, n, a, ht)
		if sdim1, rcnde1, rcndv1, iinfo, err = golapack.Dgeesx('N', sort, dslect, 'V', n, ht, wrt, wit, vs1, work, lwork, iwork, liwork, bwork); iinfo != 0 && iinfo != n+2 {
			result.Set(14, ulpinv)
			if jtype != 22 {
				fmt.Printf(" dget24: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dgeesx8", iinfo, n, jtype, iseed)
			} else {
				fmt.Printf(" dget24: %s returned info=%6d.\n         n=%6d, input example number = %4d\n", "Dgeesx8", iinfo, n, iseed[0])
			}
			info = abs(iinfo)
			goto label250
		}

		//        Perform test (15)
		if rcndv1 != rcondv {
			result.Set(14, ulpinv)
		}

		//        Perform tests (10), (11), (12), and (13)
		for i = 1; i <= n; i++ {
			if wr.Get(i-1) != wrt.Get(i-1) || wi.Get(i-1) != wit.Get(i-1) {
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

label250:
	;

	//     If there are precomputed reciprocal condition numbers, compare
	//     computed values with them.
	if comp {
		//        First set up SELOPT, SELDIM, SELVAL, SELWR, and SELWI so that
		//        the logical function Dslect selects the eigenvalues specified
		//        by NSLCT and ISLCT.
		(*seldim) = n
		(*selopt) = 1
		eps = math.Max(ulp, epsin)
		for i = 1; i <= n; i++ {
			ipnt[i-1] = i
			(*selval)[i-1] = false
			selwr.Set(i-1, wrtmp.Get(i-1))
			selwi.Set(i-1, witmp.Get(i-1))
		}
		for i = 1; i <= n-1; i++ {
			kmin = i
			vrmin = wrtmp.Get(i - 1)
			vimin = witmp.Get(i - 1)
			for j = i + 1; j <= n; j++ {
				if wrtmp.Get(j-1) < vrmin {
					kmin = j
					vrmin = wrtmp.Get(j - 1)
					vimin = witmp.Get(j - 1)
				}
			}
			wrtmp.Set(kmin-1, wrtmp.Get(i-1))
			witmp.Set(kmin-1, witmp.Get(i-1))
			wrtmp.Set(i-1, vrmin)
			witmp.Set(i-1, vimin)
			itmp = ipnt[i-1]
			ipnt[i-1] = ipnt[kmin-1]
			ipnt[kmin-1] = itmp
		}
		for i = 1; i <= nslct; i++ {
			(*selval)[ipnt[(*islct)[i-1]-1]-1] = true
		}

		//        Compute condition numbers
		golapack.Dlacpy(Full, n, n, a, ht)
		if sdim1, rconde, rcondv, iinfo, err = golapack.Dgeesx('N', 'S', dslect, 'B', n, ht, wrt, wit, vs1, work, lwork, iwork, liwork, bwork); iinfo != 0 && iinfo != n+2 {
			result.Set(15, ulpinv)
			result.Set(16, ulpinv)
			fmt.Printf(" dget24: %s returned info=%6d.\n         n=%6d, input example number = %4d\n", "Dgeesx9", iinfo, n, iseed[0])
			info = abs(iinfo)
			goto label300
		}

		//        Compare condition number for average of selected eigenvalues
		//        taking its condition number into account
		anorm = golapack.Dlange('1', n, n, a, work)
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

	label300:
	}

	return
}
