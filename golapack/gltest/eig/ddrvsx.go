package eig

import (
	"fmt"
	"math"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// ddrvsx checks the nonsymmetric eigenvalue (Schur form) problem
//    expert driver DGEESX.
//
//    ddrvsx uses both test matrices generated randomly depending on
//    data supplied in the calling sequence, as well as on data
//    read from an input file and including precomputed condition
//    numbers to which it compares the ones it computes.
//
//    When ddrvsx is called, a number of matrix "sizes" ("n's") and a
//    number of matrix "types" are specified.  For each size ("n")
//    and each type of matrix, one matrix will be generated and used
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
//    (4)     0     if WR+sqrt(-1)*WI are eigenvalues of T
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
//    (10)    0     if WR+sqrt(-1)*WI are eigenvalues of T
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
//    The "sizes" are specified by an array NN(1:NSIZES); the value of
//    each element NN(j) specifies one size.
//    The "types" are specified by a logical array DOTYPE( 1:NTYPES );
//    if DOTYPE(j) is .TRUE., then matrix type "j" will be generated.
//    Currently, the list of possible types is:
//
//    (1)  The zero matrix.
//    (2)  The identity matrix.
//    (3)  A (transposed) Jordan block, with 1's on the diagonal.
//
//    (4)  A diagonal matrix with evenly spaced entries
//         1, ..., ULP  and random signs.
//         (ULP = (first number larger than 1) - 1 )
//    (5)  A diagonal matrix with geometrically spaced entries
//         1, ..., ULP  and random signs.
//    (6)  A diagonal matrix with "clustered" entries 1, ULP, ..., ULP
//         and random signs.
//
//    (7)  Same as (4), but multiplied by a constant near
//         the overflow threshold
//    (8)  Same as (4), but multiplied by a constant near
//         the underflow threshold
//
//    (9)  A matrix of the form  U' T U, where U is orthogonal and
//         T has evenly spaced entries 1, ..., ULP with random signs
//         on the diagonal and random O(1) entries in the upper
//         triangle.
//
//    (10) A matrix of the form  U' T U, where U is orthogonal and
//         T has geometrically spaced entries 1, ..., ULP with random
//         signs on the diagonal and random O(1) entries in the upper
//         triangle.
//
//    (11) A matrix of the form  U' T U, where U is orthogonal and
//         T has "clustered" entries 1, ULP,..., ULP with random
//         signs on the diagonal and random O(1) entries in the upper
//         triangle.
//
//    (12) A matrix of the form  U' T U, where U is orthogonal and
//         T has real or complex conjugate paired eigenvalues randomly
//         chosen from ( ULP, 1 ) and random O(1) entries in the upper
//         triangle.
//
//    (13) A matrix of the form  X' T X, where X has condition
//         SQRT( ULP ) and T has evenly spaced entries 1, ..., ULP
//         with random signs on the diagonal and random O(1) entries
//         in the upper triangle.
//
//    (14) A matrix of the form  X' T X, where X has condition
//         SQRT( ULP ) and T has geometrically spaced entries
//         1, ..., ULP with random signs on the diagonal and random
//         O(1) entries in the upper triangle.
//
//    (15) A matrix of the form  X' T X, where X has condition
//         SQRT( ULP ) and T has "clustered" entries 1, ULP,..., ULP
//         with random signs on the diagonal and random O(1) entries
//         in the upper triangle.
//
//    (16) A matrix of the form  X' T X, where X has condition
//         SQRT( ULP ) and T has real or complex conjugate paired
//         eigenvalues randomly chosen from ( ULP, 1 ) and random
//         O(1) entries in the upper triangle.
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
//      computed by DGEESX and RCDEIN (the precomputed true value)
//      is supplied as input.  cond(RCONDE) is the condition number
//      of RCONDE, and takes errors in computing RCONDE into account,
//      so that the resulting quantity should be O(ULP). cond(RCONDE)
//      is essentially given by norm(A)/RCONDV.
//
//   (17)  |RCONDV - RCDVIN| / cond(RCONDV)
//
//      RCONDV is the reciprocal right invariant subspace condition
//      number computed by DGEESX and RCDVIN (the precomputed true
//      value) is supplied as input. cond(RCONDV) is the condition
//      number of RCONDV, and takes errors in computing RCONDV into
//      account, so that the resulting quantity should be O(ULP).
//      cond(RCONDV) is essentially given by norm(A)/RCONDE.
func ddrvsx(nsizes int, nn []int, ntypes int, dotype []bool, iseed []int, thresh float64, nounit int, a, h, ht *mat.Matrix, wr, wi, wrt, wit, wrtmp, witmp *mat.Vector, vs, vs1 *mat.Matrix, result, work *mat.Vector, lwork int, iwork []int, bwork []bool, t *testing.T) (err error) {
	var badnn bool
	var anorm, cond, conds, one, ovfl, rcdein, rcdvin, rtulp, rtulpi, ulp, ulpinv, unfl, zero float64
	var _i, i, iinfo, imode, itype, iwk, j, jcol, jsize, jtype, maxtyp, mtypes, n, nerrs, nfail, nmax, nnwork, nslct, ntest, ntestf, ntestt int
	adumma := make([]byte, 1)
	idumma := make([]int, 1)
	ioldsd := make([]int, 4)
	islct := make([]int, 20)
	kconds := make([]int, 21)
	kmagn := make([]int, 21)
	kmode := make([]int, 21)
	ktype := make([]int, 21)

	zero = 0.0
	one = 1.0
	maxtyp = 21

	ktype[0], ktype[1], ktype[2], ktype[3], ktype[4], ktype[5], ktype[6], ktype[7], ktype[8], ktype[9], ktype[10], ktype[11], ktype[12], ktype[13], ktype[14], ktype[15], ktype[16], ktype[17], ktype[18], ktype[19], ktype[20] = 1, 2, 3, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 9
	kmagn[0], kmagn[1], kmagn[2], kmagn[3], kmagn[4], kmagn[5], kmagn[6], kmagn[7], kmagn[8], kmagn[9], kmagn[10], kmagn[11], kmagn[12], kmagn[13], kmagn[14], kmagn[15], kmagn[16], kmagn[17], kmagn[18], kmagn[19], kmagn[20] = 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 2, 3
	kmode[0], kmode[1], kmode[2], kmode[3], kmode[4], kmode[5], kmode[6], kmode[7], kmode[8], kmode[9], kmode[10], kmode[11], kmode[12], kmode[13], kmode[14], kmode[15], kmode[16], kmode[17], kmode[18], kmode[19], kmode[20] = 0, 0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 5, 4, 3, 1, 5, 5, 5, 4, 3, 1
	kconds[0], kconds[1], kconds[2], kconds[3], kconds[4], kconds[5], kconds[6], kconds[7], kconds[8], kconds[9], kconds[10], kconds[11], kconds[12], kconds[13], kconds[14], kconds[15], kconds[16], kconds[17], kconds[18], kconds[19], kconds[20] = 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0

	path := "Dsx"

	//     Check for errors
	ntestt = 0
	ntestf = 0

	//     Important constants
	badnn = false

	//     12 is the largest dimension in the input file of precomputed
	//     problems
	nmax = 12
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
	} else if nounit <= 0 {
		err = fmt.Errorf("nounit <= 0: nounit=%v", nounit)
	} else if a.Rows < 1 || a.Rows < nmax {
		err = fmt.Errorf("a.Rows < 1 || a.Rows < nmax: a.Rows=%v, nmax=%v", a.Rows, nmax)
	} else if vs.Rows < 1 || vs.Rows < nmax {
		err = fmt.Errorf("vs.Rows < 1 || vs.Rows < nmax: vs.Rows=%v, nmax=%v", vs.Rows, nmax)
	} else if max(3*nmax, 2*pow(nmax, 2)) > lwork {
		err = fmt.Errorf("max(3*nmax, 2*pow(nmax, 2)) > lwork: nmax=%v, lwork=%v", nmax, lwork)
	}

	if err != nil {
		gltest.Xerbla2("ddrvsx", err)
		return
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

			golapack.Dlaset(Full, a.Rows, n, zero, zero, a)
			iinfo = 0
			cond = ulpinv

			//           Special Matrices -- Identity & Jordan block
			//
			//              Zero
			if itype == 1 {
				iinfo = 0

			} else if itype == 2 {
				//              Identity
				for jcol = 1; jcol <= n; jcol++ {
					a.Set(jcol-1, jcol-1, anorm)
				}

			} else if itype == 3 {
				//              Jordan Block
				for jcol = 1; jcol <= n; jcol++ {
					a.Set(jcol-1, jcol-1, anorm)
					if jcol > 1 {
						a.Set(jcol-1, jcol-1-1, one)
					}
				}

			} else if itype == 4 {
				//              Diagonal Matrix, [Eigen]values Specified
				iinfo, err = matgen.Dlatms(n, n, 'S', &iseed, 'S', work, imode, cond, anorm, 0, 0, 'N', a, work.Off(n))

			} else if itype == 5 {
				//              Symmetric, eigenvalues specified
				iinfo, err = matgen.Dlatms(n, n, 'S', &iseed, 'S', work, imode, cond, anorm, n, n, 'N', a, work.Off(n))

			} else if itype == 6 {
				//              General, eigenvalues specified
				if kconds[jtype-1] == 1 {
					conds = one
				} else if kconds[jtype-1] == 2 {
					conds = rtulpi
				} else {
					conds = zero
				}

				adumma[0] = ' '
				iinfo, err = matgen.Dlatme(n, 'S', &iseed, work, imode, cond, one, adumma, 'T', 'T', 'T', work.Off(n), 4, conds, n, n, anorm, a, work.Off(2*n))

			} else if itype == 7 {
				//              Diagonal, random eigenvalues
				iinfo, err = matgen.Dlatmr(n, n, 'S', &iseed, 'S', work, 6, one, one, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, 0, 0, zero, anorm, 'N', a, &iwork)

			} else if itype == 8 {
				//              Symmetric, random eigenvalues
				iinfo, err = matgen.Dlatmr(n, n, 'S', &iseed, 'S', work, 6, one, one, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, n, n, zero, anorm, 'N', a, &iwork)

			} else if itype == 9 {
				//              General, random eigenvalues
				iinfo, err = matgen.Dlatmr(n, n, 'S', &iseed, 'N', work, 6, one, one, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, n, n, zero, anorm, 'N', a, &iwork)
				if n >= 4 {
					golapack.Dlaset(Full, 2, n, zero, zero, a)
					golapack.Dlaset(Full, n-3, 1, zero, zero, a.Off(2, 0))
					golapack.Dlaset(Full, n-3, 2, zero, zero, a.Off(2, n-1-1))
					golapack.Dlaset(Full, 1, n, zero, zero, a.Off(n-1, 0))
				}

			} else if itype == 10 {
				//              Triangular, random eigenvalues
				iinfo, err = matgen.Dlatmr(n, n, 'S', &iseed, 'N', work, 6, one, one, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, n, 0, zero, anorm, 'N', a, &iwork)

			} else {

				iinfo = 1
			}

			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ddrvsx: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Generator", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				return
			}

		label90:
			;

			//           Test for minimal and generous workspace
			for iwk = 1; iwk <= 2; iwk++ {
				if iwk == 1 {
					nnwork = 3 * n
				} else {
					nnwork = max(3*n, 2*n*n)
				}
				nnwork = max(nnwork, 1)

				iinfo, err = dget24(false, jtype, thresh, ioldsd, nounit, n, a, h, ht, wr, wi, wrt, wit, wrtmp, witmp, vs, vs1, rcdein, rcdvin, nslct, &islct, result, work, nnwork, &iwork, &bwork)

				//              Check for RESULT(j) > THRESH
				ntest = 0
				nfail = 0
				for j = 1; j <= 15; j++ {
					if result.Get(j-1) >= zero {
						ntest = ntest + 1
					}
					if result.Get(j-1) >= thresh {
						t.Fail()
						nfail++
					}
				}

				if nfail > 0 {
					ntestf = ntestf + 1
				}
				if ntestf == 1 {
					fmt.Printf("\n %3s -- Real Schur Form Decomposition Expert Driver\n Matrix types (see ddrvsx for details):\n", path)
					fmt.Printf("\n Special Matrices:\n  1=Zero matrix.                          5=Diagonal: geometr. spaced entries.\n  2=Identity matrix.                      6=Diagonal: clustered entries.\n  3=Transposed Jordan block.              7=Diagonal: large, evenly spaced.\n  4=Diagonal: evenly spaced entries.      8=Diagonal: small, evenly spaced.\n")
					fmt.Printf(" Dense, Non-Symmetric Matrices:\n  9=Well-cond., evenly spaced eigenvals. 14=Ill-cond., geomet. spaced eigenals.\n 10=Well-cond., geom. spaced eigenvals.  15=Ill-conditioned, clustered e.vals.\n 11=Well-conditioned, clustered e.vals.  16=Ill-cond., random complex \n 12=Well-cond., random complex           17=Ill-cond., large rand. complx \n 13=Ill-conditioned, evenly spaced.      18=Ill-cond., small rand. complx \n")
					fmt.Printf(" 19=Matrix with random O(1) entries.     21=Matrix with small random entries.\n 20=Matrix with large random entries.   \n\n")
					fmt.Printf(" Tests performed with test threshold =%8.2f\n ( A denotes A on input and T denotes A on output)\n\n 1 = 0 if T in Schur form (no sort),   1/ulp otherwise\n 2 = | A - VS T transpose(VS) | / ( n |A| ulp ) (no sort)\n 3 = | I - VS transpose(VS) | / ( n ulp ) (no sort) \n 4 = 0 if WR+sqrt(-1)*WI are eigenvalues of T (no sort),  1/ulp otherwise\n 5 = 0 if T same no matter if VS computed (no sort),  1/ulp otherwise\n 6 = 0 if WR, WI same no matter if VS computed (no sort),  1/ulp otherwise\n", thresh)
					fmt.Printf(" 7 = 0 if T in Schur form (sort),   1/ulp otherwise\n 8 = | A - VS T transpose(VS) | / ( n |A| ulp ) (sort)\n 9 = | I - VS transpose(VS) | / ( n ulp ) (sort) \n 10 = 0 if WR+sqrt(-1)*WI are eigenvalues of T (sort),  1/ulp otherwise\n 11 = 0 if T same no matter what else computed (sort),  1/ulp otherwise\n 12 = 0 if WR, WI same no matter what else computed (sort), 1/ulp otherwise\n 13 = 0 if sorting successful, 1/ulp otherwise\n 14 = 0 if RCONDE same no matter what else computed, 1/ulp otherwise\n 15 = 0 if RCONDv same no matter what else computed, 1/ulp otherwise\n 16 = | RCONDE - RCONDE(precomputed) | / cond(RCONDE),\n 17 = | RCONDV - RCONDV(precomputed) | / cond(RCONDV),\n")
					ntestf = 2
				}

				for j = 1; j <= 15; j++ {
					if result.Get(j-1) >= thresh {
						t.Fail()
						fmt.Printf(" N=%5d, IWK=%2d, seed=%4d, type %2d, test(%2d)=%10.3f\n", n, iwk, ioldsd, jtype, j, result.Get(j-1))
					}
				}

				nerrs = nerrs + nfail
				ntestt = ntestt + ntest

			}
		label130:
		}
	}

	nlist := [][]int{
		{1, 1},
		{1, 1},
		{6, 6},
		{6, 0},
		{6, 6},
		{6, 1},
		{6, 3},
		{2, 1},
		{4, 2},
		{7, 6},
		{4, 2},
		{7, 5},
		{6, 4},
		{8, 4},
		{9, 3},
		{10, 4},
		{12, 6},
		{12, 7},
		{3, 1},
		{5, 1},
		{6, 4},
		{6, 2},
		{6, 3},
		{5, 1},
		{6, 2},
		{10, 1},
	}
	islctlist := [][]int{
		{1},
		{1},
		{1, 2, 3, 4, 5, 6},
		{},
		{1, 2, 3, 4, 5, 6},
		{1},
		{4, 5, 6},
		{1},
		{1, 2},
		{1, 2, 3, 4, 5, 6},
		{2, 3},
		{1, 2, 3, 4, 5},
		{3, 4, 5, 6},
		{1, 2, 3, 4},
		{1, 2, 3},
		{1, 2, 3, 4},
		{1, 2, 3, 4, 5, 6},
		{6, 7, 8, 9, 10, 11, 12},
		{1},
		{3},
		{1, 2, 3, 5},
		{3, 4},
		{3, 4, 5},
		{1},
		{1, 2},
		{1},
	}
	alist := [][]float64{
		{
			0.00000e+00,
		},
		{
			1.00000e+00,
		},
		{
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
		},
		{
			1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00,
		},
		{
			1.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 1.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 1.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 1.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00,
		},
		{
			1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			1.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 1.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 1.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 1.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 1.00000e+00,
		},
		{
			1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 2.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 3.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 4.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 5.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 6.00000e+00,
		},
		{
			1.00000e+00, 2.00000e+00,
			0.00000e+00, 3.00000e+00,
		},
		{
			8.52400e-01, 5.61100e-01, 7.04300e-01, 9.54000e-01,
			2.79800e-01, 7.21600e-01, 9.61300e-01, 3.58200e-01,
			7.08100e-01, 4.09400e-01, 2.25000e-01, 9.51800e-01,
			5.54300e-01, 5.22000e-01, 6.86000e-01, 3.07000e-02,
		},
		{
			7.81800e-01, 5.65700e-01, 7.62100e-01, 7.43600e-01, 2.55300e-01, 4.10000e-01,
			1.34000e-02,
			6.45800e-01, 2.66600e-01, 5.51000e-01, 8.31800e-01, 9.27100e-01, 6.20900e-01,
			7.83900e-01,
			1.31600e-01, 4.91400e-01, 1.77100e-01, 1.96400e-01, 1.08500e-01, 9.27000e-01,
			2.24700e-01,
			6.41000e-01, 4.68900e-01, 9.65900e-01, 8.88400e-01, 3.76900e-01, 9.67300e-01,
			6.18300e-01,
			8.38200e-01, 8.74300e-01, 4.50700e-01, 9.44200e-01, 7.75500e-01, 9.67600e-01,
			7.83100e-01,
			3.25900e-01, 7.38900e-01, 8.30200e-01, 4.52100e-01, 3.01500e-01, 2.13300e-01,
			8.43400e-01,
			5.24400e-01, 5.01600e-01, 7.52900e-01, 3.83800e-01, 8.47900e-01, 9.12800e-01,
			5.77000e-01,
		},
		{
			-9.85900e-01, 1.47840e+00, -1.33600e-01, -2.95970e+00,
			-4.33700e-01, -6.54000e-01, -7.15500e-01, 1.23760e+00,
			-7.36300e-01, -1.97680e+00, -1.95100e-01, 3.43200e-01,
			6.41400e-01, -1.40880e+00, 6.39400e-01, 8.58000e-02,
		},
		{
			2.72840e+00, 2.15200e-01, -1.05200e+00, -2.44600e-01, -6.53000e-02, 3.90500e-01,
			1.40980e+00,
			9.75300e-01, 6.51500e-01, -4.76200e-01, 5.42100e-01, 6.20900e-01, 4.75900e-01,
			-1.44930e+00,
			-9.05200e-01, 1.79000e-01, -7.08600e-01, 4.62100e-01, 1.05800e+00, 2.24260e+00,
			1.58260e+00,
			-7.17900e-01, -2.53400e-01, -4.73900e-01, -1.08100e+00, 4.13800e-01, -9.50000e-02,
			1.45300e-01,
			-1.37990e+00, -1.06490e+00, 1.25580e+00, 7.80100e-01, -6.40500e-01, -8.61000e-02,
			8.30000e-02,
			2.84900e-01, -1.29900e-01, 4.80000e-02, -2.58600e-01, 4.18900e-01, 1.37680e+00,
			8.20800e-01,
			-5.44200e-01, 9.74900e-01, 9.55800e-01, 1.23700e-01, 1.09020e+00, -1.40600e-01,
			1.90960e+00,
		},
		{
			0.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00,
			1.00000e-06, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 5.00000e-01,
		},
		{
			1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+01, 0.00000e+00,
			1.00000e+01, 0.00000e+00,
			0.00000e+00, 1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+01,
			1.00000e+01, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 1.00000e+00, -1.00000e+00, 0.00000e+00, 1.00000e+01,
			1.00000e+01, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00, 1.00000e+01,
			0.00000e+00, 1.00000e+01,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 5.00000e-01, 1.00000e+00,
			0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 5.00000e-01,
			1.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			5.00000e-01, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 5.00000e-01,
		},
		{
			1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, -1.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, -1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			7.50000e-01, 1.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 7.50000e-01, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 7.50000e-01,
		},
		{
			1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, -1.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, -1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			8.75000e-01, 1.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 8.75000e-01, 1.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 8.75000e-01, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 8.75000e-01,
		},
		{
			1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			1.00000e+01, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 1.00000e+01, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 1.00000e+01, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, -1.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+01, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, -1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+01, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+01,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			9.37500e-01, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 9.37500e-01, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 9.37500e-01, 1.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 9.37500e-01, 1.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 9.37500e-01, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 9.37500e-01,
		},
		{
			1.20000e+01, 1.10000e+01, 1.00000e+01, 9.00000e+00, 8.00000e+00, 7.00000e+00,
			6.00000e+00, 5.00000e+00, 4.00000e+00, 3.00000e+00, 2.00000e+00, 1.00000e+00,
			1.10000e+01, 1.10000e+01, 1.00000e+01, 9.00000e+00, 8.00000e+00, 7.00000e+00,
			6.00000e+00, 5.00000e+00, 4.00000e+00, 3.00000e+00, 2.00000e+00, 1.00000e+00,
			0.00000e+00, 1.00000e+01, 1.00000e+01, 9.00000e+00, 8.00000e+00, 7.00000e+00,
			6.00000e+00, 5.00000e+00, 4.00000e+00, 3.00000e+00, 2.00000e+00, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 9.00000e+00, 9.00000e+00, 8.00000e+00, 7.00000e+00,
			6.00000e+00, 5.00000e+00, 4.00000e+00, 3.00000e+00, 2.00000e+00, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 8.00000e+00, 8.00000e+00, 7.00000e+00,
			6.00000e+00, 5.00000e+00, 4.00000e+00, 3.00000e+00, 2.00000e+00, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 7.00000e+00, 7.00000e+00,
			6.00000e+00, 5.00000e+00, 4.00000e+00, 3.00000e+00, 2.00000e+00, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 6.00000e+00,
			6.00000e+00, 5.00000e+00, 4.00000e+00, 3.00000e+00, 2.00000e+00, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			5.00000e+00, 5.00000e+00, 4.00000e+00, 3.00000e+00, 2.00000e+00, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 4.00000e+00, 4.00000e+00, 3.00000e+00, 2.00000e+00, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 3.00000e+00, 3.00000e+00, 2.00000e+00, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 2.00000e+00, 2.00000e+00, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 1.00000e+00,
		},
		{
			2.00000e-06, 1.00000e+00, -2.00000e+00,
			1.00000e-06, -2.00000e+00, 4.00000e+00,
			0.00000e+00, 1.00000e+00, -2.00000e+00,
		},
		{
			2.00000e-03, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 1.00000e-03, 1.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, -1.00000e-03, 1.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, -2.00000e-03, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
		},
		{
			1.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			1.00000e+00, 1.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00,
			1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 0.00000e+00,
			1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00,
			1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00,
		},
		{
			0.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00,
			1.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 1.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00, 1.00000e+00,
			-1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00,
		},
		{
			1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00,
			5.00000e-01, 3.33300e-01, 2.50000e-01, 2.00000e-01, 1.66700e-01, 1.42900e-01,
			3.33300e-01, 2.50000e-01, 2.00000e-01, 1.66700e-01, 1.42900e-01, 1.25000e-01,
			2.50000e-01, 2.00000e-01, 1.66700e-01, 1.42900e-01, 1.25000e-01, 1.11100e-01,
			2.00000e-01, 1.66700e-01, 1.42900e-01, 1.25000e-01, 1.11100e-01, 1.00000e-01,
			1.66700e-01, 1.42900e-01, 1.25000e-01, 1.11100e-01, 1.00000e-01, 9.09000e-02,
		},
		{
			1.50000e+01, 1.10000e+01, 6.00000e+00, -9.00000e+00, -1.50000e+01,
			1.00000e+00, 3.00000e+00, 9.00000e+00, -3.00000e+00, -8.00000e+00,
			7.00000e+00, 6.00000e+00, 6.00000e+00, -3.00000e+00, -1.10000e+01,
			7.00000e+00, 7.00000e+00, 5.00000e+00, -3.00000e+00, -1.10000e+01,
			1.70000e+01, 1.20000e+01, 5.00000e+00, -1.00000e+01, -1.60000e+01,
		},
		{
			-9.00000e+00, 2.10000e+01, -1.50000e+01, 4.00000e+00, 2.00000e+00, 0.00000e+00,
			-1.00000e+01, 2.10000e+01, -1.40000e+01, 4.00000e+00, 2.00000e+00, 0.00000e+00,
			-8.00000e+00, 1.60000e+01, -1.10000e+01, 4.00000e+00, 2.00000e+00, 0.00000e+00,
			-6.00000e+00, 1.20000e+01, -9.00000e+00, 3.00000e+00, 3.00000e+00, 0.00000e+00,
			-4.00000e+00, 8.00000e+00, -6.00000e+00, 0.00000e+00, 5.00000e+00, 0.00000e+00,
			-2.00000e+00, 4.00000e+00, -3.00000e+00, 0.00000e+00, 1.00000e+00, 3.00000e+00,
		},
		{
			1.00000e+00, 1.00000e+00, 1.00000e+00, -2.00000e+00, 1.00000e+00, -1.00000e+00,
			2.00000e+00, -2.00000e+00, 4.00000e+00, -3.00000e+00,
			-1.00000e+00, 2.00000e+00, 3.00000e+00, -4.00000e+00, 2.00000e+00, -2.00000e+00,
			4.00000e+00, -4.00000e+00, 8.00000e+00, -6.00000e+00,
			-1.00000e+00, 0.00000e+00, 5.00000e+00, -5.00000e+00, 3.00000e+00, -3.00000e+00,
			6.00000e+00, -6.00000e+00, 1.20000e+01, -9.00000e+00,
			-1.00000e+00, 0.00000e+00, 3.00000e+00, -4.00000e+00, 4.00000e+00, -4.00000e+00,
			8.00000e+00, -8.00000e+00, 1.60000e+01, -1.20000e+01,
			-1.00000e+00, 0.00000e+00, 3.00000e+00, -6.00000e+00, 5.00000e+00, -4.00000e+00,
			1.00000e+01, -1.00000e+01, 2.00000e+01, -1.50000e+01,
			-1.00000e+00, 0.00000e+00, 3.00000e+00, -6.00000e+00, 2.00000e+00, -2.00000e+00,
			1.20000e+01, -1.20000e+01, 2.40000e+01, -1.80000e+01,
			-1.00000e+00, 0.00000e+00, 3.00000e+00, -6.00000e+00, 2.00000e+00, -5.00000e+00,
			1.50000e+01, -1.30000e+01, 2.80000e+01, -2.10000e+01,
			-1.00000e+00, 0.00000e+00, 3.00000e+00, -6.00000e+00, 2.00000e+00, -5.00000e+00,
			1.20000e+01, -1.10000e+01, 3.20000e+01, -2.40000e+01,
			-1.00000e+00, 0.00000e+00, 3.00000e+00, -6.00000e+00, 2.00000e+00, -5.00000e+00,
			1.20000e+01, -1.40000e+01, 3.70000e+01, -2.60000e+01,
			-1.00000e+00, 0.00000e+00, 3.00000e+00, -6.00000e+00, 2.00000e+00, -5.00000e+00,
			1.20000e+01, -1.40000e+01, 3.60000e+01, -2.50000e+01,
		},
	}
	rcdinlist := [][]float64{
		{1.00000e+00, 0.00000e+00},
		{1.00000e+00, 1.00000e+00},
		{1.00000e+00, 4.43734e-31},
		{1.00000e+00, 1.00000e+00},
		{1.00000e+00, 2.00000e+00},
		{1.00000e+00, 2.00000e+00},
		{1.00000e+00, 1.00000e+00},
		{7.07107e-01, 2.00000e+00},
		{7.22196e-01, 4.63943e-01},
		{9.43220e-01, 3.20530e+00},
		{5.22869e-01, 5.45530e-01},
		{6.04729e-01, 9.00391e-01},
		{4.89525e-05, 4.56492e-05},
		{9.56158e-05, 4.14317e-05},
		{1.00000e+00, 5.55801e-07},
		{1.00000e+00, 1.16972e-10},
		{1.85655e-10, 2.20147e-16},
		{6.92558e-05, 5.52606e-05},
		{7.30297e-01, 4.00000e+00},
		{3.99999e-12, 3.99201e-12},
		{2.93294e-01, 1.63448e-01},
		{3.97360e-01, 3.58295e-01},
		{7.28934e-01, 1.24624e-02},
		{2.17680e-01, 5.22626e-01},
		{6.78904e-02, 4.22005e-02},
		{3.60372e-02, 7.96134e-02},
	}

	//     Read in data from file to check accuracy of condition estimation
	//     Read input data until N=0
	jtype = 0

	for _i = range nlist {
		n = nlist[_i][0]
		nslct = nlist[_i][1]

		jtype = jtype + 1
		iseed[0] = jtype
		islct = islctlist[_i]
		for i = 1; i <= n; i++ {
			for j = 1; j <= n; j++ {
				a.Set(i-1, j-1, alist[_i][(i-1)*(n)+j-1])
			}
		}
		rcdein = rcdinlist[_i][0]
		rcdvin = rcdinlist[_i][1]

		iinfo, err = dget24(true, 22, thresh, iseed, nounit, n, a, h, ht, wr, wi, wrt, wit, wrtmp, witmp, vs, vs1, rcdein, rcdvin, nslct, &islct, result, work, lwork, &iwork, &bwork)

		//     Check for RESULT(j) > THRESH
		ntest = 0
		nfail = 0
		for j = 1; j <= 17; j++ {
			if result.Get(j-1) >= zero {
				ntest = ntest + 1
			}
			if result.Get(j-1) >= thresh {
				t.Fail()
				nfail++
			}
		}

		if nfail > 0 {
			ntestf = ntestf + 1
		}
		if ntestf == 1 {
			fmt.Printf("\n %3s -- Real Schur Form Decomposition Expert Driver\n Matrix types (see ddrvsx for details):\n", path)
			fmt.Printf("\n Special Matrices:\n  1=Zero matrix.                          5=Diagonal: geometr. spaced entries.\n  2=Identity matrix.                      6=Diagonal: clustered entries.\n  3=Transposed Jordan block.              7=Diagonal: large, evenly spaced.\n  4=Diagonal: evenly spaced entries.      8=Diagonal: small, evenly spaced.\n")
			fmt.Printf(" Dense, Non-Symmetric Matrices:\n  9=Well-cond., evenly spaced eigenvals. 14=Ill-cond., geomet. spaced eigenals.\n 10=Well-cond., geom. spaced eigenvals.  15=Ill-conditioned, clustered e.vals.\n 11=Well-conditioned, clustered e.vals.  16=Ill-cond., random complex \n 12=Well-cond., random complex           17=Ill-cond., large rand. complx \n 13=Ill-conditioned, evenly spaced.      18=Ill-cond., small rand. complx \n")
			fmt.Printf(" 19=Matrix with random O(1) entries.     21=Matrix with small random entries.\n 20=Matrix with large random entries.   \n\n")
			fmt.Printf(" Tests performed with test threshold =%8.2f\n ( A denotes A on input and T denotes A on output)\n\n 1 = 0 if T in Schur form (no sort),   1/ulp otherwise\n 2 = | A - VS T transpose(VS) | / ( n |A| ulp ) (no sort)\n 3 = | I - VS transpose(VS) | / ( n ulp ) (no sort) \n 4 = 0 if WR+sqrt(-1)*WI are eigenvalues of T (no sort),  1/ulp otherwise\n 5 = 0 if T same no matter if VS computed (no sort),  1/ulp otherwise\n 6 = 0 if WR, WI same no matter if VS computed (no sort),  1/ulp otherwise\n", thresh)
			fmt.Printf(" 7 = 0 if T in Schur form (sort),   1/ulp otherwise\n 8 = | A - VS T transpose(VS) | / ( n |A| ulp ) (sort)\n 9 = | I - VS transpose(VS) | / ( n ulp ) (sort) \n 10 = 0 if WR+sqrt(-1)*WI are eigenvalues of T (sort),  1/ulp otherwise\n 11 = 0 if T same no matter what else computed (sort),  1/ulp otherwise\n 12 = 0 if WR, WI same no matter what else computed (sort), 1/ulp otherwise\n 13 = 0 if sorting successful, 1/ulp otherwise\n 14 = 0 if RCONDE same no matter what else computed, 1/ulp otherwise\n 15 = 0 if RCONDv same no matter what else computed, 1/ulp otherwise\n 16 = | RCONDE - RCONDE(precomputed) | / cond(RCONDE),\n 17 = | RCONDV - RCONDV(precomputed) | / cond(RCONDV),\n")
			ntestf = 2
		}
		for j = 1; j <= 17; j++ {
			if result.Get(j-1) >= thresh {
				t.Fail()
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
