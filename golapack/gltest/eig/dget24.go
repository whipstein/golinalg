package eig

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dget24 checks the nonsymmetric eigenvalue (Schur form) problem
//    expert driver DGEESX.
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
//       computed by DGEESX and RCDEIN (the precomputed true value)
//       is supplied as input.  cond(RCONDE) is the condition number
//       of RCONDE, and takes errors in computing RCONDE into account,
//       so that the resulting quantity should be O(ULP). cond(RCONDE)
//       is essentially given by norm(A)/RCONDV.
//
//    (17)  |RCONDV - RCDVIN| / cond(RCONDV)
//
//       RCONDV is the reciprocal right invariant subspace condition
//       number computed by DGEESX and RCDVIN (the precomputed true
//       value) is supplied as input. cond(RCONDV) is the condition
//       number of RCONDV, and takes errors in computing RCONDV into
//       account, so that the resulting quantity should be O(ULP).
//       cond(RCONDV) is essentially given by norm(A)/RCONDE.
func Dget24(comp bool, jtype *int, thresh *float64, iseed *[]int, nounit *int, n *int, a *mat.Matrix, lda *int, h, ht *mat.Matrix, wr, wi, wrt, wit, wrtmp, witmp *mat.Vector, vs *mat.Matrix, ldvs *int, vs1 *mat.Matrix, rcdein, rcdvin *float64, nslct *int, islct *[]int, result, work *mat.Vector, lwork *int, iwork *[]int, bwork *[]bool, info *int) {
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
		(*info) = -18
	} else if (*lwork) < 3*(*n) {
		(*info) = -26
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DGET24"), -(*info))
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
	liwork = (*n) * (*n)
	for isort = 0; isort <= 1; isort++ {
		if isort == 0 {
			sort = 'N'
			rsub = 0
		} else {
			sort = 'S'
			rsub = 6
		}

		//        Compute Schur form and Schur vectors, and test them
		golapack.Dlacpy('F', n, n, a, lda, h, lda)
		golapack.Dgeesx('V', sort, Dslect, 'N', n, h, lda, &sdim, wr, wi, vs, ldvs, &rconde, &rcondv, work, lwork, iwork, &liwork, bwork, &iinfo)
		if iinfo != 0 && iinfo != (*n)+2 {
			result.Set(1+rsub-1, ulpinv)
			if (*jtype) != 22 {
				fmt.Printf(" DGET24: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DGEESX1", iinfo, *n, *jtype, *iseed)
			} else {
				fmt.Printf(" DGET24: %s returned INFO=%6d.\n         N=%6d, INPUT EXAMPLE NUMBER = %4d\n", "DGEESX1", iinfo, *n, (*iseed)[0])
			}
			(*info) = absint(iinfo)
			return
		}
		if isort == 0 {
			goblas.Dcopy(n, wr, func() *int { y := 1; return &y }(), wrtmp, func() *int { y := 1; return &y }())
			goblas.Dcopy(n, wi, func() *int { y := 1; return &y }(), witmp, func() *int { y := 1; return &y }())
		}

		//        Do Test (1) or Test (7)
		result.Set(1+rsub-1, zero)
		for j = 1; j <= (*n)-2; j++ {
			for i = j + 2; i <= (*n); i++ {
				if h.Get(i-1, j-1) != zero {
					result.Set(1+rsub-1, ulpinv)
				}
			}
		}
		for i = 1; i <= (*n)-2; i++ {
			if h.Get(i+1-1, i-1) != zero && h.Get(i+2-1, i+1-1) != zero {
				result.Set(1+rsub-1, ulpinv)
			}
		}
		for i = 1; i <= (*n)-1; i++ {
			if h.Get(i+1-1, i-1) != zero {
				if h.Get(i-1, i-1) != h.Get(i+1-1, i+1-1) || h.Get(i-1, i+1-1) == zero || math.Copysign(one, h.Get(i+1-1, i-1)) == math.Copysign(one, h.Get(i-1, i+1-1)) {
					result.Set(1+rsub-1, ulpinv)
				}
			}
		}

		//        Test (2) or (8): Compute norm(A - Q*H*Q') / (norm(A) * N * ULP)
		//
		//        Copy A to VS1, used as workspace
		golapack.Dlacpy(' ', n, n, a, lda, vs1, ldvs)

		//        Compute Q*H and store in HT.
		goblas.Dgemm(NoTrans, NoTrans, n, n, n, &one, vs, ldvs, h, lda, &zero, ht, lda)

		//        Compute A - Q*H*Q'
		goblas.Dgemm(NoTrans, Trans, n, n, n, toPtrf64(-one), ht, lda, vs, ldvs, &one, vs1, ldvs)

		anorm = maxf64(golapack.Dlange('1', n, n, a, lda, work), smlnum)
		wnorm = golapack.Dlange('1', n, n, vs1, ldvs, work)

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
		Dort01('C', n, n, vs, ldvs, work, lwork, result.GetPtr(3+rsub-1))

		//        Do Test (4) or Test (10)
		result.Set(4+rsub-1, zero)
		for i = 1; i <= (*n); i++ {
			if h.Get(i-1, i-1) != wr.Get(i-1) {
				result.Set(4+rsub-1, ulpinv)
			}
		}
		if (*n) > 1 {
			if h.Get(1, 0) == zero && wi.Get(0) != zero {
				result.Set(4+rsub-1, ulpinv)
			}
			if h.Get((*n)-1, (*n)-1-1) == zero && wi.Get((*n)-1) != zero {
				result.Set(4+rsub-1, ulpinv)
			}
		}
		for i = 1; i <= (*n)-1; i++ {
			if h.Get(i+1-1, i-1) != zero {
				tmp = math.Sqrt(math.Abs(h.Get(i+1-1, i-1))) * math.Sqrt(math.Abs(h.Get(i-1, i+1-1)))
				result.Set(4+rsub-1, maxf64(result.Get(4+rsub-1), math.Abs(wi.Get(i-1)-tmp)/maxf64(ulp*tmp, smlnum)))
				result.Set(4+rsub-1, maxf64(result.Get(4+rsub-1), math.Abs(wi.Get(i+1-1)+tmp)/maxf64(ulp*tmp, smlnum)))
			} else if i > 1 {
				if h.Get(i+1-1, i-1) == zero && h.Get(i-1, i-1-1) == zero && wi.Get(i-1) != zero {
					result.Set(4+rsub-1, ulpinv)
				}
			}
		}

		//        Do Test (5) or Test (11)
		golapack.Dlacpy('F', n, n, a, lda, ht, lda)
		golapack.Dgeesx('N', sort, Dslect, 'N', n, ht, lda, &sdim, wrt, wit, vs, ldvs, &rconde, &rcondv, work, lwork, iwork, &liwork, bwork, &iinfo)
		if iinfo != 0 && iinfo != (*n)+2 {
			result.Set(5+rsub-1, ulpinv)
			if (*jtype) != 22 {
				fmt.Printf(" DGET24: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DGEESX2", iinfo, *n, *jtype, *iseed)
			} else {
				fmt.Printf(" DGET24: %s returned INFO=%6d.\n         N=%6d, INPUT EXAMPLE NUMBER = %4d\n", "DGEESX2", iinfo, *n, (*iseed)[0])
			}
			(*info) = absint(iinfo)
			goto label250
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
			if wr.Get(i-1) != wrt.Get(i-1) || wi.Get(i-1) != wit.Get(i-1) {
				result.Set(6+rsub-1, ulpinv)
			}
		}

		//        Do Test (13)
		if isort == 1 {
			result.Set(12, zero)
			knteig = 0
			for i = 1; i <= (*n); i++ {
				if Dslect(wr.GetPtr(i-1), wi.GetPtr(i-1)) || Dslect(wr.GetPtr(i-1), toPtrf64(-wi.Get(i-1))) {
					knteig = knteig + 1
				}
				if i < (*n) {
					if (Dslect(wr.GetPtr(i+1-1), wi.GetPtr(i+1-1)) || Dslect(wr.GetPtr(i+1-1), toPtrf64(-wi.Get(i+1-1)))) && (!(Dslect(wr.GetPtr(i-1), wi.GetPtr(i-1)) || Dslect(wr.GetPtr(i-1), toPtrf64(-wi.Get(i-1))))) && iinfo != (*n)+2 {
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
	if (*lwork) >= (*n)+((*n)*(*n))/2 {
		//        Compute both RCONDE and RCONDV with VS
		sort = 'S'
		result.Set(13, zero)
		result.Set(14, zero)
		golapack.Dlacpy('F', n, n, a, lda, ht, lda)
		golapack.Dgeesx('V', sort, Dslect, 'B', n, ht, lda, &sdim1, wrt, wit, vs1, ldvs, &rconde, &rcondv, work, lwork, iwork, &liwork, bwork, &iinfo)
		if iinfo != 0 && iinfo != (*n)+2 {
			result.Set(13, ulpinv)
			result.Set(14, ulpinv)
			if (*jtype) != 22 {
				fmt.Printf(" DGET24: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DGEESX3", iinfo, *n, *jtype, *iseed)
			} else {
				fmt.Printf(" DGET24: %s returned INFO=%6d.\n         N=%6d, INPUT EXAMPLE NUMBER = %4d\n", "DGEESX3", iinfo, *n, (*iseed)[0])
			}
			(*info) = absint(iinfo)
			goto label250
		}

		//        Perform tests (10), (11), (12), and (13)
		for i = 1; i <= (*n); i++ {
			if wr.Get(i-1) != wrt.Get(i-1) || wi.Get(i-1) != wit.Get(i-1) {
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
		golapack.Dlacpy('F', n, n, a, lda, ht, lda)
		golapack.Dgeesx('N', sort, Dslect, 'B', n, ht, lda, &sdim1, wrt, wit, vs1, ldvs, &rcnde1, &rcndv1, work, lwork, iwork, &liwork, bwork, &iinfo)
		if iinfo != 0 && iinfo != (*n)+2 {
			result.Set(13, ulpinv)
			result.Set(14, ulpinv)
			if (*jtype) != 22 {
				fmt.Printf(" DGET24: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DGEESX4", iinfo, *n, *jtype, *iseed)
			} else {
				fmt.Printf(" DGET24: %s returned INFO=%6d.\n         N=%6d, INPUT EXAMPLE NUMBER = %4d\n", "DGEESX4", iinfo, *n, (*iseed)[0])
			}
			(*info) = absint(iinfo)
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
		for i = 1; i <= (*n); i++ {
			if wr.Get(i-1) != wrt.Get(i-1) || wi.Get(i-1) != wit.Get(i-1) {
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
		golapack.Dlacpy('F', n, n, a, lda, ht, lda)
		golapack.Dgeesx('V', sort, Dslect, 'E', n, ht, lda, &sdim1, wrt, wit, vs1, ldvs, &rcnde1, &rcndv1, work, lwork, iwork, &liwork, bwork, &iinfo)
		if iinfo != 0 && iinfo != (*n)+2 {
			result.Set(13, ulpinv)
			if (*jtype) != 22 {
				fmt.Printf(" DGET24: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DGEESX5", iinfo, *n, *jtype, *iseed)
			} else {
				fmt.Printf(" DGET24: %s returned INFO=%6d.\n         N=%6d, INPUT EXAMPLE NUMBER = %4d\n", "DGEESX5", iinfo, *n, (*iseed)[0])
			}
			(*info) = absint(iinfo)
			goto label250
		}

		//        Perform test (14)
		if rcnde1 != rconde {
			result.Set(13, ulpinv)
		}

		//        Perform tests (10), (11), (12), and (13)
		for i = 1; i <= (*n); i++ {
			if wr.Get(i-1) != wrt.Get(i-1) || wi.Get(i-1) != wit.Get(i-1) {
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
		golapack.Dlacpy('F', n, n, a, lda, ht, lda)
		golapack.Dgeesx('N', sort, Dslect, 'E', n, ht, lda, &sdim1, wrt, wit, vs1, ldvs, &rcnde1, &rcndv1, work, lwork, iwork, &liwork, bwork, &iinfo)
		if iinfo != 0 && iinfo != (*n)+2 {
			result.Set(13, ulpinv)
			if (*jtype) != 22 {
				fmt.Printf(" DGET24: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DGEESX6", iinfo, *n, *jtype, *iseed)
			} else {
				fmt.Printf(" DGET24: %s returned INFO=%6d.\n         N=%6d, INPUT EXAMPLE NUMBER = %4d\n", "DGEESX6", iinfo, *n, (*iseed)[0])
			}
			(*info) = absint(iinfo)
			goto label250
		}

		//        Perform test (14)
		if rcnde1 != rconde {
			result.Set(13, ulpinv)
		}

		//        Perform tests (10), (11), (12), and (13)
		for i = 1; i <= (*n); i++ {
			if wr.Get(i-1) != wrt.Get(i-1) || wi.Get(i-1) != wit.Get(i-1) {
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
		golapack.Dlacpy('F', n, n, a, lda, ht, lda)
		golapack.Dgeesx('V', sort, Dslect, 'V', n, ht, lda, &sdim1, wrt, wit, vs1, ldvs, &rcnde1, &rcndv1, work, lwork, iwork, &liwork, bwork, &iinfo)
		if iinfo != 0 && iinfo != (*n)+2 {
			result.Set(14, ulpinv)
			if (*jtype) != 22 {
				fmt.Printf(" DGET24: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DGEESX7", iinfo, *n, *jtype, *iseed)
			} else {
				fmt.Printf(" DGET24: %s returned INFO=%6d.\n         N=%6d, INPUT EXAMPLE NUMBER = %4d\n", "DGEESX7", iinfo, *n, (*iseed)[0])
			}
			(*info) = absint(iinfo)
			goto label250
		}

		//        Perform test (15)
		if rcndv1 != rcondv {
			result.Set(14, ulpinv)
		}

		//        Perform tests (10), (11), (12), and (13)
		for i = 1; i <= (*n); i++ {
			if wr.Get(i-1) != wrt.Get(i-1) || wi.Get(i-1) != wit.Get(i-1) {
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
		golapack.Dlacpy('F', n, n, a, lda, ht, lda)
		golapack.Dgeesx('N', sort, Dslect, 'V', n, ht, lda, &sdim1, wrt, wit, vs1, ldvs, &rcnde1, &rcndv1, work, lwork, iwork, &liwork, bwork, &iinfo)
		if iinfo != 0 && iinfo != (*n)+2 {
			result.Set(14, ulpinv)
			if (*jtype) != 22 {
				fmt.Printf(" DGET24: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DGEESX8", iinfo, *n, *jtype, *iseed)
			} else {
				fmt.Printf(" DGET24: %s returned INFO=%6d.\n         N=%6d, INPUT EXAMPLE NUMBER = %4d\n", "DGEESX8", iinfo, *n, (*iseed)[0])
			}
			(*info) = absint(iinfo)
			goto label250
		}

		//        Perform test (15)
		if rcndv1 != rcondv {
			result.Set(14, ulpinv)
		}

		//        Perform tests (10), (11), (12), and (13)
		for i = 1; i <= (*n); i++ {
			if wr.Get(i-1) != wrt.Get(i-1) || wi.Get(i-1) != wit.Get(i-1) {
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

label250:
	;

	//     If there are precomputed reciprocal condition numbers, compare
	//     computed values with them.
	if comp {
		//        First set up SELOPT, SELDIM, SELVAL, SELWR, and SELWI so that
		//        the logical function Dslect selects the eigenvalues specified
		//        by NSLCT and ISLCT.
		(*seldim) = (*n)
		(*selopt) = 1
		eps = maxf64(ulp, epsin)
		for i = 1; i <= (*n); i++ {
			ipnt[i-1] = i
			(*selval)[i-1] = false
			selwr.Set(i-1, wrtmp.Get(i-1))
			selwi.Set(i-1, witmp.Get(i-1))
		}
		for i = 1; i <= (*n)-1; i++ {
			kmin = i
			vrmin = wrtmp.Get(i - 1)
			vimin = witmp.Get(i - 1)
			for j = i + 1; j <= (*n); j++ {
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
		for i = 1; i <= (*nslct); i++ {
			(*selval)[ipnt[(*islct)[i-1]-1]-1] = true
		}

		//        Compute condition numbers
		golapack.Dlacpy('F', n, n, a, lda, ht, lda)
		golapack.Dgeesx('N', 'S', Dslect, 'B', n, ht, lda, &sdim1, wrt, wit, vs1, ldvs, &rconde, &rcondv, work, lwork, iwork, &liwork, bwork, &iinfo)
		if iinfo != 0 && iinfo != (*n)+2 {
			result.Set(15, ulpinv)
			result.Set(16, ulpinv)
			fmt.Printf(" DGET24: %s returned INFO=%6d.\n         N=%6d, INPUT EXAMPLE NUMBER = %4d\n", "DGEESX9", iinfo, *n, (*iseed)[0])
			(*info) = absint(iinfo)
			goto label300
		}

		//        Compare condition number for average of selected eigenvalues
		//        taking its condition number into account
		anorm = golapack.Dlange('1', n, n, a, lda, work)
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

	label300:
	}
}
