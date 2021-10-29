package eig

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// zdrvsx checks the nonsymmetric eigenvalue (Schur form) problem
//    expert driver ZGEESX.
//
//    zdrvsx uses both test matrices generated randomly depending on
//    data supplied in the calling sequence, as well as on data
//    read from an input file and including precomputed condition
//    numbers to which it compares the ones it computes.
//
//    When zdrvsx is called, a number of matrix "sizes" ("n's") and a
//    number of matrix "types" are specified.  For each size ("n")
//    and each _type of matrix, one matrix will be generated and used
//    to test the nonsymmetric eigenroutines.  For each matrix, 15
//    tests will be performed:
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
//    The "sizes" are specified by an array NN(1:NSIZES); the value of
//    each element NN(j) specifies one size.
//    The "types" are specified by a logical array DOTYPE( 1:NTYPES );
//    if DOTYPE(j) is .TRUE., then matrix _type "j" will be generated.
//    Currently, the list of possible types is:
//
//    (1)  The zero matrix.
//    (2)  The identity matrix.
//    (3)  A (transposed) Jordan block, with 1's on the diagonal.
//
//    (4)  A diagonal matrix with evenly spaced entries
//         1, ..., ULP  and random complex angles.
//         (ULP = (first number larger than 1) - 1 )
//    (5)  A diagonal matrix with geometrically spaced entries
//         1, ..., ULP  and random complex angles.
//    (6)  A diagonal matrix with "clustered" entries 1, ULP, ..., ULP
//         and random complex angles.
//
//    (7)  Same as (4), but multiplied by a constant near
//         the overflow threshold
//    (8)  Same as (4), but multiplied by a constant near
//         the underflow threshold
//
//    (9)  A matrix of the form  U' T U, where U is unitary and
//         T has evenly spaced entries 1, ..., ULP with random
//         complex angles on the diagonal and random O(1) entries in
//         the upper triangle.
//
//    (10) A matrix of the form  U' T U, where U is unitary and
//         T has geometrically spaced entries 1, ..., ULP with random
//         complex angles on the diagonal and random O(1) entries in
//         the upper triangle.
//
//    (11) A matrix of the form  U' T U, where U is orthogonal and
//         T has "clustered" entries 1, ULP,..., ULP with random
//         complex angles on the diagonal and random O(1) entries in
//         the upper triangle.
//
//    (12) A matrix of the form  U' T U, where U is unitary and
//         T has complex eigenvalues randomly chosen from
//         ULP < |z| < 1   and random O(1) entries in the upper
//         triangle.
//
//    (13) A matrix of the form  X' T X, where X has condition
//         SQRT( ULP ) and T has evenly spaced entries 1, ..., ULP
//         with random complex angles on the diagonal and random O(1)
//         entries in the upper triangle.
//
//    (14) A matrix of the form  X' T X, where X has condition
//         SQRT( ULP ) and T has geometrically spaced entries
//         1, ..., ULP with random complex angles on the diagonal
//         and random O(1) entries in the upper triangle.
//
//    (15) A matrix of the form  X' T X, where X has condition
//         SQRT( ULP ) and T has "clustered" entries 1, ULP,..., ULP
//         with random complex angles on the diagonal and random O(1)
//         entries in the upper triangle.
//
//    (16) A matrix of the form  X' T X, where X has condition
//         SQRT( ULP ) and T has complex eigenvalues randomly chosen
//         from ULP < |z| < 1 and random O(1) entries in the upper
//         triangle.
//
//    (17) Same as (16), but multiplied by a constant
//         near the overflow threshold
//    (18) Same as (16), but multiplied by a constant
//         near the underflow threshold
//
//    (19) Nonsymmetric matrix with random entries chosen from (-1,1).
//         If N is at least 4, all entries in first two rows and last
//         row, and first column and last two columns are zero.
//    (20) Same as (19), but multiplied by a constant
//         near the overflow threshold
//    (21) Same as (19), but multiplied by a constant
//         near the underflow threshold
//
//    In addition, an input file will be read from logical unit number
//    NIUNIT. The file contains matrices along with precomputed
//    eigenvalues and reciprocal condition numbers for the eigenvalue
//    average and right invariant subspace. For these matrices, in
//    addition to tests (1) to (15) we will compute the following two
//    tests:
//
//   (16)  |RCONDE - RCDEIN| / cond(RCONDE)
//
//      RCONDE is the reciprocal average eigenvalue condition number
//      computed by ZGEESX and RCDEIN (the precomputed true value)
//      is supplied as input.  cond(RCONDE) is the condition number
//      of RCONDE, and takes errors in computing RCONDE into account,
//      so that the resulting quantity should be O(ULP). cond(RCONDE)
//      is essentially given by norm(A)/RCONDV.
//
//   (17)  |RCONDV - RCDVIN| / cond(RCONDV)
//
//      RCONDV is the reciprocal right invariant subspace condition
//      number computed by ZGEESX and RCDVIN (the precomputed true
//      value) is supplied as input. cond(RCONDV) is the condition
//      number of RCONDV, and takes errors in computing RCONDV into
//      account, so that the resulting quantity should be O(ULP).
//      cond(RCONDV) is essentially given by norm(A)/RCONDE.
func zdrvsx(nsizes int, nn []int, ntypes int, dotype []bool, iseed []int, thresh float64, a, h, ht *mat.CMatrix, w, wt, wtmp *mat.CVector, vs, vs1 *mat.CMatrix, result *mat.Vector, work *mat.CVector, lwork int, rwork *mat.Vector, bwork []bool) (err error) {
	var badnn bool
	var cone, czero complex128
	var anorm, cond, conds, one, ovfl, rcdein, rcdvin, rtulp, rtulpi, ulp, ulpinv, unfl, zero float64
	var _i, i, iinfo, imode, isrt, itype, iwk, j, jcol, jsize, jtype, maxtyp, mtypes, n, nerrs, nfail, nmax, nnwork, nslct, ntest, ntestf, ntestt int
	idumma := make([]int, 1)
	ioldsd := make([]int, 4)
	islct := make([]int, 20)
	kconds := []int{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0}
	kmagn := []int{1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 2, 3}
	kmode := []int{0, 0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 5, 4, 3, 1, 5, 5, 5, 4, 3, 1}
	ktype := []int{1, 2, 3, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 9}

	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	zero = 0.0
	one = 1.0
	maxtyp = 21

	path := "Zsx"

	//     Check for errors
	ntestt = 0
	ntestf = 0

	//     Important constants
	badnn = false

	//     8 is the largest dimension in the input file of precomputed
	//     problems
	nmax = 8
	for j = 1; j <= nsizes; j++ {
		nmax = max(nmax, nn[j-1])
		if nn[j-1] < 0 {
			badnn = true
		}
	}

	//     Check for errors
	if nsizes < 0 {
		err = fmt.Errorf("nsizes < 0: nsizes=%v", nsizes)
	} else if badnn {
		err = fmt.Errorf("badnn: nn=%v", nn)
	} else if ntypes < 0 {
		err = fmt.Errorf("ntypes < 0: ntypes=%v", ntypes)
	} else if thresh < zero {
		err = fmt.Errorf("thresh < zero: thresh=%v", thresh)
	} else if a.Rows < 1 || a.Rows < nmax {
		err = fmt.Errorf("a.Rows < 1 || a.Rows < nmax: a.Rows=%v, nmax=%v", a.Rows, nmax)
	} else if vs.Rows < 1 || vs.Rows < nmax {
		err = fmt.Errorf("vs.Rows < 1 || vs.Rows < nmax: vs.Rows=%v, nmax=%v", vs.Rows, nmax)
	} else if max(3*nmax, 2*pow(nmax, 2)) > lwork {
		err = fmt.Errorf("max(3*nmax, 2*pow(nmax, 2)) > lwork: nmax=%v, lwork=%v", nmax, lwork)
	}

	if err != nil {
		gltest.Xerbla2("zdrvsx", err)
		return
	}

	//     If nothing to do check on NIUNIT
	if nsizes == 0 || ntypes == 0 {
		goto label150
	}

	//     More Important constants
	unfl = golapack.Dlamch(SafeMinimum)
	ovfl = one / unfl
	unfl, ovfl = golapack.Dlabad(unfl, ovfl)
	ulp = golapack.Dlamch(Precision)
	ulpinv = one / ulp
	rtulp = math.Sqrt(ulp)
	rtulpi = one / rtulp

	//     Loop over sizes, types
	nerrs = 0

	for jsize = 1; jsize <= nsizes; jsize++ {
		n = nn[jsize-1]
		if nsizes != 1 {
			mtypes = min(maxtyp, ntypes)
		} else {
			mtypes = min(maxtyp+1, ntypes)
		}

		for jtype = 1; jtype <= mtypes; jtype++ {
			if !dotype[jtype-1] {
				goto label130
			}

			//           Save ISEED in case of an error.
			for j = 1; j <= 4; j++ {
				ioldsd[j-1] = iseed[j-1]
			}

			//           Compute "A"
			//
			//           Control parameters:
			//
			//           KMAGN  KCONDS  KMODE        KTYPE
			//       =1  O(1)   1       clustered 1  zero
			//       =2  large  large   clustered 2  identity
			//       =3  small          exponential  Jordan
			//       =4                 arithmetic   diagonal, (w/ eigenvalues)
			//       =5                 random log   symmetric, w/ eigenvalues
			//       =6                 random       general, w/ eigenvalues
			//       =7                              random diagonal
			//       =8                              random symmetric
			//       =9                              random general
			//       =10                             random triangular
			if mtypes > maxtyp {
				goto label90
			}

			itype = ktype[jtype-1]
			imode = kmode[jtype-1]

			//           Compute norm
			switch kmagn[jtype-1] {
			case 1:
				goto label30
			case 2:
				goto label40
			case 3:
				goto label50
			}

		label30:
			;
			anorm = one
			goto label60

		label40:
			;
			anorm = ovfl * ulp
			goto label60

		label50:
			;
			anorm = unfl * ulpinv
			goto label60

		label60:
			;

			golapack.Zlaset(Full, a.Rows, n, czero, czero, a)
			iinfo = 0
			cond = ulpinv

			//           Special Matrices -- Identity & Jordan block
			if itype == 1 {
				//              Zero
				iinfo = 0

			} else if itype == 2 {
				//              Identity
				for jcol = 1; jcol <= n; jcol++ {
					a.SetRe(jcol-1, jcol-1, anorm)
				}

			} else if itype == 3 {
				//              Jordan Block
				for jcol = 1; jcol <= n; jcol++ {
					a.SetRe(jcol-1, jcol-1, anorm)
					if jcol > 1 {
						a.Set(jcol-1, jcol-1-1, cone)
					}
				}

			} else if itype == 4 {
				//              Diagonal Matrix, [Eigen]values Specified
				err = matgen.Zlatms(n, n, 'S', &iseed, 'H', rwork, imode, cond, anorm, 0, 0, 'N', a, work.Off(n))

			} else if itype == 5 {
				//              Symmetric, eigenvalues specified
				err = matgen.Zlatms(n, n, 'S', &iseed, 'H', rwork, imode, cond, anorm, n, n, 'N', a, work.Off(n))

			} else if itype == 6 {
				//              General, eigenvalues specified
				if kconds[jtype-1] == 1 {
					conds = one
				} else if kconds[jtype-1] == 2 {
					conds = rtulpi
				} else {
					conds = zero
				}

				err = matgen.Zlatme(n, 'D', &iseed, work, imode, cond, cone, 'T', 'T', 'T', rwork, 4, conds, n, n, anorm, a, work.Off(2*n))

			} else if itype == 7 {
				//              Diagonal, random eigenvalues
				err = matgen.Zlatmr(n, n, 'D', &iseed, 'N', work, 6, one, cone, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, 0, 0, zero, anorm, 'N', a, &idumma)

			} else if itype == 8 {
				//              Symmetric, random eigenvalues
				err = matgen.Zlatmr(n, n, 'D', &iseed, 'H', work, 6, one, cone, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, n, n, zero, anorm, 'N', a, &idumma)

			} else if itype == 9 {
				//              General, random eigenvalues
				err = matgen.Zlatmr(n, n, 'D', &iseed, 'N', work, 6, one, cone, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, n, n, zero, anorm, 'N', a, &idumma)
				if n >= 4 {
					golapack.Zlaset(Full, 2, n, czero, czero, a)
					golapack.Zlaset(Full, n-3, 1, czero, czero, a.Off(2, 0))
					golapack.Zlaset(Full, n-3, 2, czero, czero, a.Off(2, n-1-1))
					golapack.Zlaset(Full, 1, n, czero, czero, a.Off(n-1, 0))
				}

			} else if itype == 10 {
				//              Triangular, random eigenvalues
				err = matgen.Zlatmr(n, n, 'D', &iseed, 'N', work, 6, one, cone, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, n, 0, zero, anorm, 'N', a, &idumma)

			} else {

				iinfo = 1
			}

			if iinfo != 0 || err != nil {
				fmt.Printf(" zdrvsx: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Generator", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				return
			}

		label90:
			;

			//           Test for minimal and generous workspace
			for iwk = 1; iwk <= 2; iwk++ {
				if iwk == 1 {
					nnwork = 2 * n
				} else {
					nnwork = max(2*n, n*(n+1)/2)
				}
				nnwork = max(nnwork, 1)

				err = zget24(false, jtype, thresh, ioldsd, n, a, h, ht, w, wt, wtmp, vs, vs1, rcdein, rcdvin, nslct, islct, 0, result, work, nnwork, rwork, &bwork)

				//              Check for RESULT(j) > THRESH
				ntest = 0
				nfail = 0
				for j = 1; j <= 15; j++ {
					if result.Get(j-1) >= zero {
						ntest++
					}
					if result.Get(j-1) >= thresh {
						nfail++
					}
				}

				if nfail > 0 {
					ntestf++
				}
				if ntestf == 1 {
					fmt.Printf("\n %3s -- Complex Schur Form Decomposition Expert Driver\n Matrix types (see zdrvsx for details): \n", path)
					fmt.Printf("\n Special Matrices:\n  1=Zero matrix.                          5=Diagonal: geometr. spaced entries.\n  2=Identity matrix.                      6=Diagonal: clustered entries.\n  3=Transposed Jordan block.              7=Diagonal: large, evenly spaced.\n  4=Diagonal: evenly spaced entries.      8=Diagonal: small, evenly spaced.\n")
					fmt.Printf(" Dense, Non-Symmetric Matrices:\n  9=Well-cond., evenly spaced eigenvals. 14=Ill-cond., geomet. spaced eigenals.\n 10=Well-cond., geom. spaced eigenvals.  15=Ill-conditioned, clustered e.vals.\n 11=Well-conditioned, clustered e.vals.  16=Ill-cond., random complex \n 12=Well-cond., random complex           17=Ill-cond., large rand. complx \n 13=Ill-conditioned, evenly spaced.      18=Ill-cond., small rand. complx \n")
					fmt.Printf(" 19=Matrix with random O(1) entries.     21=Matrix with small random entries.\n 20=Matrix with large random entries.   \n\n")
					fmt.Printf(" Tests performed with test threshold =%8.2f\n ( A denotes A on input and T denotes A on output)\n\n 1 = 0 if T in Schur form (no sort),   1/ulp otherwise\n 2 = | A - VS T transpose(VS) | / ( n |A| ulp ) (no sort)\n 3 = | I - VS transpose(VS) | / ( n ulp ) (no sort) \n 4 = 0 if W are eigenvalues of T (no sort),  1/ulp otherwise\n 5 = 0 if T same no matter if VS computed (no sort),  1/ulp otherwise\n 6 = 0 if W same no matter if VS computed (no sort),  1/ulp otherwise\n", thresh)
					fmt.Printf(" 7 = 0 if T in Schur form (sort),   1/ulp otherwise\n 8 = | A - VS T transpose(VS) | / ( n |A| ulp ) (sort)\n 9 = | I - VS transpose(VS) | / ( n ulp ) (sort) \n 10 = 0 if W are eigenvalues of T (sort),  1/ulp otherwise\n 11 = 0 if T same no matter what else computed (sort),  1/ulp otherwise\n 12 = 0 if W same no matter what else computed (sort), 1/ulp otherwise\n 13 = 0 if sorting successful, 1/ulp otherwise\n 14 = 0 if RCONDE same no matter what else computed, 1/ulp otherwise\n 15 = 0 if RCONDv same no matter what else computed, 1/ulp otherwise\n 16 = | RCONDE - RCONDE(precomputed) | / cond(RCONDE),\n 17 = | RCONDV - RCONDV(precomputed) | / cond(RCONDV),\n")
					ntestf = 2
				}

				for j = 1; j <= 15; j++ {
					if result.Get(j-1) >= thresh {
						fmt.Printf(" N=%5d, IWK=%2d, seed=%4d, _type %2d, test(%2d)=%10.3f\n", n, iwk, ioldsd, jtype, j, result.Get(j-1))
						err = fmt.Errorf(" N=%5d, IWK=%2d, seed=%4d, _type %2d, test(%2d)=%10.3f\n", n, iwk, ioldsd, jtype, j, result.Get(j-1))
					}
				}

				nerrs = nerrs + nfail
				ntestt = ntestt + ntest

			}
		label130:
		}
	}

label150:
	;

	//     Read in data from file to check accuracy of condition estimation
	//     Read input data until N=0
	jtype = 0
	nlist := []int{1, 1, 5, 5, 5, 6, 6, 4, 4, 3, 4, 4, 4, 5, 3, 4, 7, 5, 8, 3}
	nslctlist := []int{1, 1, 3, 3, 2, 3, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 4, 3, 4, 2}
	isrtlist := []int{0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0}
	islctlist := [][]int{
		{1},
		{1},
		{2, 3, 4},
		{1, 3, 5},
		{2, 4},
		{3, 4, 6},
		{1, 3, 5},
		{3, 4},
		{2, 3},
		{2, 3},
		{1, 3},
		{1, 3, 4},
		{2, 3},
		{2, 3},
		{1, 2},
		{1, 3},
		{1, 4, 6, 7},
		{1, 3, 5},
		{1, 2, 3, 4},
		{2, 3},
	}
	alist := [][]complex128{
		{
			0.0000e+00 + 0.0000e+00i,
		},
		{
			1.0000e+00 + 0.0000e+00i,
		},
		{
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
		},
		{
			1.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 1.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 1.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 1.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 1.0000e+00 + 0.0000e+00i,
		},
		{
			1.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 2.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 3.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 4.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 5.0000e+00 + 0.0000e+00i,
		},
		{
			0.0000e+00 + 1.0000e+00i, 1.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 1.0000e+00i, 1.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 1.0000e+00i, 1.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 1.0000e+00i, 1.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 1.0000e+00i, 1.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 1.0000e+00i,
		},
		{
			0.0000e+00 + 1.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			1.0000e+00 + 0.0000e+00i, 0.0000e+00 + 1.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 1.0000e+00 + 0.0000e+00i, 0.0000e+00 + 1.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 1.0000e+00 + 0.0000e+00i, 0.0000e+00 + 1.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 1.0000e+00 + 0.0000e+00i, 0.0000e+00 + 1.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 1.0000e+00 + 0.0000e+00i, 0.0000e+00 + 1.0000e+00i,
		},
		{
			9.4480e-01 + 1.0000e+00i, 6.7670e-01 + 1.0000e+00i, 6.9080e-01 + 1.0000e+00i, 5.9650e-01 + 1.0000e+00i,
			5.8760e-01 + 1.0000e+00i, 8.6420e-01 + 1.0000e+00i, 6.7690e-01 + 1.0000e+00i, 7.2600e-02 + 1.0000e+00i,
			7.2560e-01 + 1.0000e+00i, 1.9430e-01 + 1.0000e+00i, 9.6870e-01 + 1.0000e+00i, 2.8310e-01 + 1.0000e+00i,
			2.8490e-01 + 1.0000e+00i, 5.8000e-02 + 1.0000e+00i, 4.8450e-01 + 1.0000e+00i, 7.3610e-01 + 1.0000e+00i,
		},
		{
			2.1130e-01 + 9.9330e-01i, 8.0960e-01 + 4.2370e-01i, 4.8320e-01 + 1.1670e-01i, 6.5380e-01 + 4.9430e-01i,
			8.2400e-02 + 8.3600e-01i, 8.4740e-01 + 2.6130e-01i, 6.1350e-01 + 6.2500e-01i, 4.8990e-01 + 3.6500e-02i,
			7.5990e-01 + 7.4690e-01i, 4.5240e-01 + 2.4030e-01i, 2.7490e-01 + 5.5100e-01i, 7.7410e-01 + 2.2600e-01i,
			8.7000e-03 + 3.7800e-02i, 8.0750e-01 + 3.4050e-01i, 8.8070e-01 + 3.5500e-01i, 9.6260e-01 + 8.1590e-01i,
		},
		{
			1.0000e+00 + 2.0000e+00i, 3.0000e+00 + 4.0000e+00i, 2.1000e+01 + 2.2000e+01i,
			4.3000e+01 + 4.4000e+01i, 1.3000e+01 + 1.4000e+01i, 1.5000e+01 + 1.6000e+01i,
			5.0000e+00 + 6.0000e+00i, 7.0000e+00 + 8.0000e+00i, 2.5000e+01 + 2.6000e+01i,
		},
		{
			5.0000e+00 + 9.0000e+00i, 5.0000e+00 + 5.0000e+00i, -6.0000e+00 + -6.0000e+00i, -7.0000e+00 + -7.0000e+00i,
			3.0000e+00 + 3.0000e+00i, 6.0000e+00 + 1.0000e+01i, -5.0000e+00 + -5.0000e+00i, -6.0000e+00 + -6.0000e+00i,
			2.0000e+00 + 2.0000e+00i, 3.0000e+00 + 3.0000e+00i, -1.0000e+00 + 3.0000e+00i, -5.0000e+00 + -5.0000e+00i,
			1.0000e+00 + 1.0000e+00i, 2.0000e+00 + 2.0000e+00i, -3.0000e+00 + -3.0000e+00i, 0.0000e+00 + 4.0000e+00i,
		},
		{
			3.0000e+00 + 0.0000e+00i, 1.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 2.0000e+00i,
			1.0000e+00 + 0.0000e+00i, 3.0000e+00 + 0.0000e+00i, 0.0000e+00 + -2.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 2.0000e+00i, 1.0000e+00 + 0.0000e+00i, 1.0000e+00 + 0.0000e+00i,
			0.0000e+00 + -2.0000e+00i, 0.0000e+00 + 0.0000e+00i, 1.0000e+00 + 0.0000e+00i, 1.0000e+00 + 0.0000e+00i,
		},
		{
			7.0000e+00 + 0.0000e+00i, 3.0000e+00 + 0.0000e+00i, 1.0000e+00 + 2.0000e+00i, -1.0000e+00 + 2.0000e+00i,
			3.0000e+00 + 0.0000e+00i, 7.0000e+00 + 0.0000e+00i, 1.0000e+00 + -2.0000e+00i, -1.0000e+00 + -2.0000e+00i,
			1.0000e+00 + -2.0000e+00i, 1.0000e+00 + 2.0000e+00i, 7.0000e+00 + 0.0000e+00i, -3.0000e+00 + 0.0000e+00i,
			-1.0000e+00 + -2.0000e+00i, -2.0000e+00 + 2.0000e+00i, -3.0000e+00 + 0.0000e+00i, 7.0000e+00 + 0.0000e+00i,
		},
		{
			1.0000e+00 + 2.0000e+00i, 3.0000e+00 + 4.0000e+00i, 2.1000e+01 + 2.2000e+01i, 2.3000e+01 + 2.4000e+01i, 4.1000e+01 + 4.2000e+01i,
			4.3000e+01 + 4.4000e+01i, 1.3000e+01 + 1.4000e+01i, 1.5000e+01 + 1.6000e+01i, 3.3000e+01 + 3.4000e+01i, 3.5000e+01 + 3.6000e+01i,
			5.0000e+00 + 6.0000e+00i, 7.0000e+00 + 8.0000e+00i, 2.5000e+01 + 2.6000e+01i, 2.7000e+01 + 2.8000e+01i, 4.5000e+01 + 4.6000e+01i,
			4.7000e+01 + 4.8000e+01i, 1.7000e+01 + 1.8000e+01i, 1.9000e+01 + 2.0000e+01i, 3.7000e+01 + 3.8000e+01i, 3.9000e+01 + 4.0000e+01i,
			9.0000e+00 + 1.0000e+01i, 1.1000e+01 + 1.2000e+01i, 2.9000e+01 + 3.0000e+01i, 3.1000e+01 + 3.2000e+01i, 4.9000e+01 + 5.0000e+01i,
		},
		{
			1.0000e+00 + 1.0000e+00i, -1.0000e+00 + -1.0000e+00i, 2.0000e+00 + 2.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 1.0000e+00i, 2.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, -1.0000e+00 + 0.0000e+00i, 3.0000e+00 + 1.0000e+00i,
		},
		{
			-4.0000e+00 + -2.0000e+00i, -5.0000e+00 + -6.0000e+00i, -2.0000e+00 + -6.0000e+00i, 0.0000e+00 + -2.0000e+00i,
			1.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 1.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 1.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
		},
		{
			2.0000e+00 + 4.0000e+00i, 1.0000e+00 + 1.0000e+00i, 6.0000e+00 + 2.0000e+00i, 3.0000e+00 + 3.0000e+00i, 5.0000e+00 + 5.0000e+00i, 2.0000e+00 + 6.0000e+00i, 1.0000e+00 + 1.0000e+00i,
			1.0000e+00 + 2.0000e+00i, 1.0000e+00 + 3.0000e+00i, 3.0000e+00 + 1.0000e+00i, 5.0000e+00 + -4.0000e+00i, 1.0000e+00 + 1.0000e+00i, 7.0000e+00 + 2.0000e+00i, 2.0000e+00 + 3.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 3.0000e+00 + -2.0000e+00i, 1.0000e+00 + 1.0000e+00i, 6.0000e+00 + 3.0000e+00i, 2.0000e+00 + 1.0000e+00i, 1.0000e+00 + 4.0000e+00i, 2.0000e+00 + 1.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 2.0000e+00 + 3.0000e+00i, 3.0000e+00 + 1.0000e+00i, 1.0000e+00 + 2.0000e+00i, 2.0000e+00 + 2.0000e+00i, 3.0000e+00 + 1.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 2.0000e+00 + -1.0000e+00i, 2.0000e+00 + 2.0000e+00i, 3.0000e+00 + 1.0000e+00i, 1.0000e+00 + 3.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 1.0000e+00 + -1.0000e+00i, 2.0000e+00 + 1.0000e+00i, 2.0000e+00 + 2.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 2.0000e+00 + -2.0000e+00i, 1.0000e+00 + 1.0000e+00i,
		},
		{
			0.0000e+00 + 5.0000e+00i, 1.0000e+00 + 2.0000e+00i, 2.0000e+00 + 3.0000e+00i, -3.0000e+00 + 6.0000e+00i, 6.0000e+00 + 0.0000e+00i,
			-1.0000e+00 + 2.0000e+00i, 0.0000e+00 + 6.0000e+00i, 4.0000e+00 + 5.0000e+00i, -3.0000e+00 + -2.0000e+00i, 5.0000e+00 + 0.0000e+00i,
			-2.0000e+00 + 3.0000e+00i, -4.0000e+00 + 5.0000e+00i, 0.0000e+00 + 7.0000e+00i, 3.0000e+00 + 0.0000e+00i, 2.0000e+00 + 0.0000e+00i,
			3.0000e+00 + 6.0000e+00i, 3.0000e+00 + -2.0000e+00i, -3.0000e+00 + 0.0000e+00i, 0.0000e+00 + -5.0000e+00i, 2.0000e+00 + 1.0000e+00i,
			-6.0000e+00 + 0.0000e+00i, -5.0000e+00 + 0.0000e+00i, -2.0000e+00 + 0.0000e+00i, -2.0000e+00 + 1.0000e+00i, 0.0000e+00 + 2.0000e+00i,
		},
		{
			0.0000e+00 + 1.0000e+00i, 1.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 1.0000e+00i, 1.0000e+00 + 0.0000e+00i, 0.0000e+00 + 1.0000e+00i, 1.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 1.0000e+00i, 1.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 2.0000e+00i, 2.0000e+00 + 0.0000e+00i, 0.0000e+00 + 2.0000e+00i, 2.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 1.0000e+00i, 1.0000e+00 + 0.0000e+00i, 0.0000e+00 + 3.0000e+00i, 3.0000e+00 + 0.0000e+00i, 0.0000e+00 + 3.0000e+00i, 3.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 1.0000e+00i, 0.0000e+00 + 4.0000e+00i, 4.0000e+00 + 0.0000e+00i, 0.0000e+00 + 4.0000e+00i, 4.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 9.5000e-01i, 1.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 9.5000e-01i, 1.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 9.5000e-01i, 1.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 9.5000e-01i,
		},
		{
			2.0000e+00 + 0.0000e+00i, 0.0000e+00 + -1.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 1.0000e+00i, 2.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 3.0000e+00 + 0.0000e+00i,
		},
	}
	rlist := [][]float64{
		{1.0000e+00, 0.0000e+00},
		{1.0000e+00, 1.0000e+00},
		{1.0000e+00, 2.9582e-31},
		{1.0000e+00, 1.0000e+00},
		{1.0000e+00, 1.0000e+00},
		{1.0000e+00, 2.0000e+00},
		{1.0000e+00, 2.0000e+00},
		{9.6350e-01, 3.3122e-01},
		{8.4053e-01, 7.4754e-01},
		{3.9550e-01, 2.0464e+01},
		{3.3333e-01, 1.2569e-01},
		{1.0000e+00, 8.2843e-01},
		{9.8985e-01, 4.1447e+00},
		{3.1088e-01, 4.6912e+00},
		{2.2361e-01, 1.0000e+00},
		{7.2803e-05, 1.1947e-04},
		{3.7241e-01, 5.2080e-01},
		{1.0000e+00, 4.5989e+00},
		{9.5269e-12, 2.9360e-11},
		{1.0000e+00, 2.0000e+00},
	}

	for _i, n = range nlist {
		nslct = nslctlist[_i]
		isrt = isrtlist[_i]

		jtype = jtype + 1
		iseed[0] = jtype
		for i = 1; i <= nslct; i++ {
			islct[i-1] = islctlist[_i][i-1]
		}
		for i = 1; i <= n; i++ {
			for j = 1; j <= n; j++ {
				a.Set(i-1, j-1, alist[_i][(i-1)*(n)+j-1])
			}
		}
		rcdein = rlist[_i][0]
		rcdvin = rlist[_i][1]

		err = zget24(true, 22, thresh, iseed, n, a, h, ht, w, wt, wtmp, vs, vs1, rcdein, rcdvin, nslct, islct, isrt, result, work, lwork, rwork, &bwork)

		//     Check for RESULT(j) > THRESH
		ntest = 0
		nfail = 0
		for j = 1; j <= 17; j++ {
			if result.Get(j-1) >= zero {
				ntest = ntest + 1
			}
			if result.Get(j-1) >= thresh {
				nfail++
			}
		}

		if nfail > 0 {
			ntestf = ntestf + 1
		}
		if ntestf == 1 {
			fmt.Printf("\n %3s -- Complex Schur Form Decomposition Expert Driver\n Matrix types (see zdrvsx for details): \n", path)
			fmt.Printf("\n Special Matrices:\n  1=Zero matrix.                          5=Diagonal: geometr. spaced entries.\n  2=Identity matrix.                      6=Diagonal: clustered entries.\n  3=Transposed Jordan block.              7=Diagonal: large, evenly spaced.\n  4=Diagonal: evenly spaced entries.      8=Diagonal: small, evenly spaced.\n")
			fmt.Printf(" Dense, Non-Symmetric Matrices:\n  9=Well-cond., evenly spaced eigenvals. 14=Ill-cond., geomet. spaced eigenals.\n 10=Well-cond., geom. spaced eigenvals.  15=Ill-conditioned, clustered e.vals.\n 11=Well-conditioned, clustered e.vals.  16=Ill-cond., random complex \n 12=Well-cond., random complex           17=Ill-cond., large rand. complx \n 13=Ill-conditioned, evenly spaced.      18=Ill-cond., small rand. complx \n")
			fmt.Printf(" 19=Matrix with random O(1) entries.     21=Matrix with small random entries.\n 20=Matrix with large random entries.   \n\n")
			fmt.Printf(" Tests performed with test threshold =%8.2f\n ( A denotes A on input and T denotes A on output)\n\n 1 = 0 if T in Schur form (no sort),   1/ulp otherwise\n 2 = | A - VS T transpose(VS) | / ( n |A| ulp ) (no sort)\n 3 = | I - VS transpose(VS) | / ( n ulp ) (no sort) \n 4 = 0 if W are eigenvalues of T (no sort),  1/ulp otherwise\n 5 = 0 if T same no matter if VS computed (no sort),  1/ulp otherwise\n 6 = 0 if W same no matter if VS computed (no sort),  1/ulp otherwise\n", thresh)
			fmt.Printf(" 7 = 0 if T in Schur form (sort),   1/ulp otherwise\n 8 = | A - VS T transpose(VS) | / ( n |A| ulp ) (sort)\n 9 = | I - VS transpose(VS) | / ( n ulp ) (sort) \n 10 = 0 if W are eigenvalues of T (sort),  1/ulp otherwise\n 11 = 0 if T same no matter what else computed (sort),  1/ulp otherwise\n 12 = 0 if W same no matter what else computed (sort), 1/ulp otherwise\n 13 = 0 if sorting successful, 1/ulp otherwise\n 14 = 0 if RCONDE same no matter what else computed, 1/ulp otherwise\n 15 = 0 if RCONDv same no matter what else computed, 1/ulp otherwise\n 16 = | RCONDE - RCONDE(precomputed) | / cond(RCONDE),\n 17 = | RCONDV - RCONDV(precomputed) | / cond(RCONDV),\n")
			ntestf = 2
		}
		for j = 1; j <= 17; j++ {
			if result.Get(j-1) >= thresh {
				fmt.Printf(" N=%5d, input example =%3d,  test(%2d)=%10.3f\n", n, jtype, j, result.Get(j-1))
			}
		}

		nerrs = nerrs + nfail
		ntestt = ntestt + ntest
	}

	//     Summary
	dlasum(path, nerrs, ntestt)

	return
}
