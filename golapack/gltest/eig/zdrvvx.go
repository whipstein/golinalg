package eig

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// zdrvvx checks the nonsymmetric eigenvalue problem expert driver
//    ZGEEVX.
//
//    zdrvvx uses both test matrices generated randomly depending on
//    data supplied in the calling sequence, as well as on data
//    read from an input file and including precomputed condition
//    numbers to which it compares the ones it computes.
//
//    When zdrvvx is called, a number of matrix "sizes" ("n's") and a
//    number of matrix "types" are specified in the calling sequence.
//    For each size ("n") and each _type of matrix, one matrix will be
//    generated and used to test the nonsymmetric eigenroutines.  For
//    each matrix, 9 tests will be performed:
//
//    (1)     | A * VR - VR * W | / ( n |A| ulp )
//
//      Here VR is the matrix of unit right eigenvectors.
//      W is a diagonal matrix with diagonal entries W(j).
//
//    (2)     | A**H  * VL - VL * W**H | / ( n |A| ulp )
//
//      Here VL is the matrix of unit left eigenvectors, A**H is the
//      conjugate transpose of A, and W is as above.
//
//    (3)     | |VR(i)| - 1 | / ulp and largest component real
//
//      VR(i) denotes the i-th column of VR.
//
//    (4)     | |VL(i)| - 1 | / ulp and largest component real
//
//      VL(i) denotes the i-th column of VL.
//
//    (5)     W(full) = W(partial)
//
//      W(full) denotes the eigenvalues computed when VR, VL, RCONDV
//      and RCONDE are also computed, and W(partial) denotes the
//      eigenvalues computed when only some of VR, VL, RCONDV, and
//      RCONDE are computed.
//
//    (6)     VR(full) = VR(partial)
//
//      VR(full) denotes the right eigenvectors computed when VL, RCONDV
//      and RCONDE are computed, and VR(partial) denotes the result
//      when only some of VL and RCONDV are computed.
//
//    (7)     VL(full) = VL(partial)
//
//      VL(full) denotes the left eigenvectors computed when VR, RCONDV
//      and RCONDE are computed, and VL(partial) denotes the result
//      when only some of VR and RCONDV are computed.
//
//    (8)     0 if SCALE, ILO, IHI, ABNRM (full) =
//                 SCALE, ILO, IHI, ABNRM (partial)
//            1/ulp otherwise
//
//      SCALE, ILO, IHI and ABNRM describe how the matrix is balanced.
//      (full) is when VR, VL, RCONDE and RCONDV are also computed, and
//      (partial) is when some are not computed.
//
//    (9)     RCONDV(full) = RCONDV(partial)
//
//      RCONDV(full) denotes the reciprocal condition numbers of the
//      right eigenvectors computed when VR, VL and RCONDE are also
//      computed. RCONDV(partial) denotes the reciprocal condition
//      numbers when only some of VR, VL and RCONDE are computed.
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
//         T has evenly spaced entries 1, ..., ULP with random complex
//         angles on the diagonal and random O(1) entries in the upper
//         triangle.
//
//    (10) A matrix of the form  U' T U, where U is unitary and
//         T has geometrically spaced entries 1, ..., ULP with random
//         complex angles on the diagonal and random O(1) entries in
//         the upper triangle.
//
//    (11) A matrix of the form  U' T U, where U is unitary and
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
//    (19) Nonsymmetric matrix with random entries chosen from |z| < 1
//         If N is at least 4, all entries in first two rows and last
//         row, and first column and last two columns are zero.
//    (20) Same as (19), but multiplied by a constant
//         near the overflow threshold
//    (21) Same as (19), but multiplied by a constant
//         near the underflow threshold
//
//    In addition, an input file will be read from logical unit number
//    NIUNIT. The file contains matrices along with precomputed
//    eigenvalues and reciprocal condition numbers for the eigenvalues
//    and right eigenvectors. For these matrices, in addition to tests
//    (1) to (9) we will compute the following two tests:
//
//   (10)  |RCONDV - RCDVIN| / cond(RCONDV)
//
//      RCONDV is the reciprocal right eigenvector condition number
//      computed by ZGEEVX and RCDVIN (the precomputed true value)
//      is supplied as input. cond(RCONDV) is the condition number of
//      RCONDV, and takes errors in computing RCONDV into account, so
//      that the resulting quantity should be O(ULP). cond(RCONDV) is
//      essentially given by norm(A)/RCONDE.
//
//   (11)  |RCONDE - RCDEIN| / cond(RCONDE)
//
//      RCONDE is the reciprocal eigenvalue condition number
//      computed by ZGEEVX and RCDEIN (the precomputed true value)
//      is supplied as input.  cond(RCONDE) is the condition number
//      of RCONDE, and takes errors in computing RCONDE into account,
//      so that the resulting quantity should be O(ULP). cond(RCONDE)
//      is essentially given by norm(A)/RCONDV.
func zdrvvx(nsizes int, nn []int, ntypes int, dotype []bool, iseed []int, thresh float64, a, h *mat.CMatrix, w, w1 *mat.CVector, vl, vr, lre *mat.CMatrix, rcondv, rcndv1, rcdvin, rconde, rcnde1, rcdein, scale, scale1, result *mat.Vector, work *mat.CVector, nwork int, rwork *mat.Vector) (err error) {
	var badnn bool
	var balanc byte
	var cone, czero complex128
	var anorm, cond, conds, one, ovfl, rtulp, rtulpi, ulp, ulpinv, unfl, wi, wr, zero float64
	var _i, i, ibal, iinfo, imode, isrt, itype, iwk, j, jcol, jsize, jtype, maxtyp, mtypes, n, nerrs, nfail, nmax, nnwork, ntest, ntestf, ntestt int
	bal := []byte{'N', 'P', 'S', 'B'}
	idumma := make([]int, 1)
	ioldsd := make([]int, 4)
	kconds := []int{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0}
	kmagn := []int{1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 2, 3}
	kmode := []int{0, 0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 5, 4, 3, 1, 5, 5, 5, 4, 3, 1}
	ktype := []int{1, 2, 3, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 9}

	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	zero = 0.0
	one = 1.0
	maxtyp = 21

	path := "Zvx"

	//     Check for errors
	ntestt = 0
	ntestf = 0

	//     Important constants
	badnn = false

	//     7 is the largest dimension in the input file of precomputed
	//     problems
	nmax = 7
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
	} else if vl.Rows < 1 || vl.Rows < nmax {
		err = fmt.Errorf("vl.Rows < 1 || vl.Rows < nmax: vl.Rows=%v, nmax=%v", vl.Rows, nmax)
	} else if vr.Rows < 1 || vr.Rows < nmax {
		err = fmt.Errorf("vr.Rows < 1 || vr.Rows < nmax: vr.Rows=%v, nmax=%v", vr.Rows, nmax)
	} else if lre.Rows < 1 || lre.Rows < nmax {
		err = fmt.Errorf("lre.Rows < 1 || lre.Rows < nmax: lre.Rows=%v, nmax=%v", lre.Rows, nmax)
	} else if 6*nmax+2*pow(nmax, 2) > nwork {
		err = fmt.Errorf("6*nmax+2*pow(nmax, 2) > nwork: nmax=%v, nwork=%v", nmax, nwork)
	}

	if err != nil {
		gltest.Xerbla2("zdrvvx", err)
		return
	}

	//     If nothing to do check on NIUNIT
	if nsizes == 0 || ntypes == 0 {
		goto label160
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
				goto label140
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
			//
			//              Zero
			if itype == 1 {
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
						a.SetRe(jcol-1, jcol-1-1, one)
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
				err = matgen.Zlatmr(n, n, 'D', &iseed, 'S', work, 6, one, cone, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, 0, 0, zero, anorm, 'N', a, &idumma)

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
				fmt.Printf(" zdrvvx: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Generator", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				return
			}

		label90:
			;

			//           Test for minimal and generous workspace
			for iwk = 1; iwk <= 3; iwk++ {
				if iwk == 1 {
					nnwork = 2 * n
				} else if iwk == 2 {
					nnwork = 2*n + pow(n, 2)
				} else {
					nnwork = 6*n + 2*pow(n, 2)
				}
				nnwork = max(nnwork, 1)

				//              Test for all balancing options
				for ibal = 1; ibal <= 4; ibal++ {
					balanc = bal[ibal-1]

					//                 Perform tests
					err = zget23(false, 0, balanc, jtype, thresh, ioldsd, n, a, h, w, w1, vl, vr, lre, rcondv, rcndv1, rcdvin, rconde, rcnde1, rcdein, scale, scale1, result, work, nnwork, rwork)

					//                 Check for RESULT(j) > THRESH
					ntest = 0
					nfail = 0
					for j = 1; j <= 9; j++ {
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
						fmt.Printf("\n %3s -- Complex Eigenvalue-Eigenvector Decomposition Expert Driver\n Matrix types (see zdrvvx for details): \n", path)
						fmt.Printf("\n Special Matrices:\n  1=Zero matrix.                          5=Diagonal: geometr. spaced entries.\n  2=Identity matrix.                      6=Diagonal: clustered entries.\n  3=Transposed Jordan block.              7=Diagonal: large, evenly spaced.\n  4=Diagonal: evenly spaced entries.      8=Diagonal: small, evenly spaced.\n")
						fmt.Printf(" Dense, Non-Symmetric Matrices:\n  9=Well-cond., evenly spaced eigenvals. 14=Ill-cond., geomet. spaced eigenals.\n 10=Well-cond., geom. spaced eigenvals.  15=Ill-conditioned, clustered e.vals.\n 11=Well-conditioned, clustered e.vals.  16=Ill-cond., random complex \n 12=Well-cond., random complex           17=Ill-cond., large rand. complx \n 13=Ill-conditioned, evenly spaced.      18=Ill-cond., small rand. complx \n")
						fmt.Printf(" 19=Matrix with random O(1) entries.     21=Matrix with small random entries.\n 20=Matrix with large random entries.    22=Matrix read from input file\n\n")
						fmt.Printf(" Tests performed with test threshold =%8.2f\n\n 1 = | A VR - VR W | / ( n |A| ulp ) \n 2 = | transpose(A) VL - VL W | / ( n |A| ulp ) \n 3 = | |VR(i)| - 1 | / ulp \n 4 = | |VL(i)| - 1 | / ulp \n 5 = 0 if W same no matter if VR or VL computed, 1/ulp otherwise\n 6 = 0 if VR same no matter what else computed,  1/ulp otherwise\n 7 = 0 if VL same no matter what else computed,  1/ulp otherwise\n 8 = 0 if RCONDV same no matter what else computed,  1/ulp otherwise\n 9 = 0 if SCALE, ILO, IHI, ABNRM same no matter what else computed,  1/ulp otherwise\n 10 = | RCONDV - RCONDV(precomputed) | / cond(RCONDV),\n 11 = | RCONDE - RCONDE(precomputed) | / cond(RCONDE),\n", thresh)
						ntestf = 2
					}
					//
					for j = 1; j <= 9; j++ {
						if result.Get(j-1) >= thresh {
							fmt.Printf(" BALANC='%c',N=%4d,IWK=%1d, seed=%4d, _type %2d, test(%2d)=%10.3f\n", balanc, n, iwk, ioldsd, jtype, j, result.Get(j-1))
							err = fmt.Errorf(" BALANC='%c',N=%4d,IWK=%1d, seed=%4d, _type %2d, test(%2d)=%10.3f\n", balanc, n, iwk, ioldsd, jtype, j, result.Get(j-1))
						}
					}

					nerrs = nerrs + nfail
					ntestt = ntestt + ntest

				}
			}
		label140:
		}
	}

label160:
	;

	//     Read in data from file to check accuracy of condition estimation.
	//     Assume input eigenvalues are sorted lexicographically (increasing
	//     by real part, then decreasing by imaginary part)
	jtype = 0

	nlist := []int{1, 1, 2, 2, 2, 5, 5, 5, 6, 6, 4, 4, 3, 4, 4, 4, 5, 3, 4, 7, 5, 3}
	isrtlist := []int{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0}
	alist := [][]complex128{
		{
			0.0000e+00 + 0.0000e+00i,
		},
		{
			0.0000e+00 + 1.0000e+00i,
		},
		{
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
		},
		{
			3.0000e+00 + 0.0000e+00i, 2.0000e+00 + 0.0000e+00i,
			2.0000e+00 + 0.0000e+00i, 3.0000e+00 + 0.0000e+00i,
		},
		{
			3.0000e+00 + 0.0000e+00i, 0.0000e+00 + 2.0000e+00i,
			0.0000e+00 + 2.0000e+00i, 3.0000e+00 + 0.0000e+00i,
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
			2.0000e+00 + 0.0000e+00i, 0.0000e+00 + -1.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 1.0000e+00i, 2.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 3.0000e+00 + 0.0000e+00i,
		},
	}
	wlist := [][]float64{
		{
			0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
		},
		{
			0.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
		},
		{
			0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
		},
		{
			1.0000e+00, 0.0000e+00, 1.0000e+00, 4.0000e+00,
			5.0000e+00, 0.0000e+00, 1.0000e+00, 4.0000e+00,
		},
		{
			3.0000e+00, 2.0000e+00, 1.0000e+00, 4.0000e+00,
			3.0000e+00, -2.0000e+00, 1.0000e+00, 4.0000e+00,
		},
		{
			0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
		},
		{
			1.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
			1.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
			1.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
			1.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
			1.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
		},
		{
			1.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00,
			2.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00,
			3.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00,
			4.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00,
			5.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00,
		},
		{
			0.0000e+00, 1.0000e+00, 1.1921e-07, 0.0000e+00,
			0.0000e+00, 1.0000e+00, 2.4074e-35, 0.0000e+00,
			0.0000e+00, 1.0000e+00, 2.4074e-35, 0.0000e+00,
			0.0000e+00, 1.0000e+00, 2.4074e-35, 0.0000e+00,
			0.0000e+00, 1.0000e+00, 2.4074e-35, 0.0000e+00,
			0.0000e+00, 1.0000e+00, 1.1921e-07, 0.0000e+00,
		},
		{
			0.0000e+00, 1.0000e+00, 1.1921e-07, 0.0000e+00,
			0.0000e+00, 1.0000e+00, 2.4074e-35, 0.0000e+00,
			0.0000e+00, 1.0000e+00, 2.4074e-35, 0.0000e+00,
			0.0000e+00, 1.0000e+00, 2.4074e-35, 0.0000e+00,
			0.0000e+00, 1.0000e+00, 2.4074e-35, 0.0000e+00,
			0.0000e+00, 1.0000e+00, 1.1921e-07, 0.0000e+00,
		},
		{
			2.6014e-01, -1.7813e-01, 8.5279e-01, 3.2881e-01,
			2.8961e-01, 2.0772e-01, 8.4871e-01, 3.2358e-01,
			7.3990e-01, -4.6522e-04, 9.7398e-01, 3.4994e-01,
			2.2242e+00, 3.9709e+00, 9.8325e-01, 4.1429e+00,
		},
		{
			-6.2157e-01, 6.0607e-01, 8.7533e-01, 8.1980e-01,
			2.8890e-01, -2.6354e-01, 8.2538e-01, 8.1086e-01,
			3.8017e-01, 5.4217e-01, 7.4771e-01, 7.0323e-01,
			2.2487e+00, 1.7368e+00, 9.2372e-01, 2.2178e+00,
		},
		{
			-7.4775e+00, 6.8803e+00, 3.9550e-01, 1.6583e+01,
			6.7009e+00, -7.8760e+00, 3.9828e-01, 1.6312e+01,
			3.9777e+01, 4.2996e+01, 7.9686e-01, 3.7399e+01,
		},
		{
			1.0000e+00, 5.0000e+00, 2.1822e-01, 7.4651e-01,
			2.0000e+00, 6.0000e+00, 2.1822e-01, 3.0893e-01,
			3.0000e+00, 7.0000e+00, 2.1822e-01, 1.8315e-01,
			4.0000e+00, 8.0000e+00, 2.1822e-01, 6.6350e-01,
		},
		{
			-8.2843e-01, 1.6979e-07, 1.0000e+00, 8.2843e-01,
			4.1744e-07, 7.1526e-08, 1.0000e+00, 8.2843e-01,
			4.0000e+00, 1.6690e-07, 1.0000e+00, 8.2843e-01,
			4.8284e+00, 6.8633e-08, 1.0000e+00, 8.2843e-01,
		},
		{
			-8.0767e-03, -2.5211e-01, 9.9864e-01, 7.7961e+00,
			7.7723e+00, 2.4349e-01, 7.0272e-01, 3.3337e-01,
			8.0000e+00, -3.4273e-07, 7.0711e-01, 3.3337e-01,
			1.2236e+01, 8.6188e-03, 9.9021e-01, 3.9429e+00,
		},
		{
			-9.4600e+00, 7.2802e+00, 3.1053e-01, 1.1937e+01,
			-7.7912e-06, -1.2743e-05, 2.9408e-01, 1.6030e-05,
			-7.3042e-06, 3.2789e-06, 7.2259e-01, 6.7794e-06,
			7.0733e+00, -9.5584e+00, 3.0911e-01, 1.1891e+01,
			1.2739e+02, 1.3228e+02, 9.2770e-01, 1.2111e+02,
		},
		{
			1.0000e+00, 1.0000e+00, 3.0151e-01, 0.0000e+00,
			1.0000e+00, 1.0000e+00, 3.1623e-01, 0.0000e+00,
			2.0000e+00, 1.0000e+00, 2.2361e-01, 1.0000e+00,
		},
		{
			-9.9883e-01, -1.0006e+00, 1.3180e-04, 2.4106e-04,
			-1.0012e+00, -9.9945e-01, 1.3140e-04, 2.4041e-04,
			-9.9947e-01, -6.8325e-04, 1.3989e-04, 8.7487e-05,
			-1.0005e+00, 6.8556e-04, 1.4010e-04, 8.7750e-05,
		},
		{
			-2.7081e+00, -2.8029e+00, 6.9734e-01, 3.9279e+00,
			-1.1478e+00, 8.0176e-01, 6.5772e-01, 9.4243e-01,
			-8.0109e-01, 4.9694e+00, 4.6751e-01, 1.3779e+00,
			9.9492e-01, 3.1688e+00, 3.5095e-01, 5.9845e-01,
			2.0809e+00, 1.9341e+00, 4.9042e-01, 3.9035e-01,
			5.3138e+00, 1.2242e+00, 3.0213e-01, 7.1268e-01,
			8.2674e+00, 3.7047e+00, 2.8270e-01, 3.2849e+00,
		},
		{
			-4.1735e-08, -1.0734e+01, 1.0000e+00, 7.7345e+00,
			-2.6397e-07, -2.9991e+00, 1.0000e+00, 4.5989e+00,
			1.4565e-07, 1.5998e+00, 1.0000e+00, 4.5989e+00,
			-4.4369e-07, 9.3159e+00, 1.0000e+00, 7.7161e+00,
			4.0937e-09, 1.7817e+01, 1.0000e+00, 8.5013e+00,
		},
		{
			1.0000e+00, 0.0000e+00, 1.0000e+00, 2.0000e+00,
			3.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
			3.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
		},
	}

	for _i, n = range nlist {
		isrt = isrtlist[_i]
		//     Read input data until N=0
		jtype = jtype + 1
		iseed[0] = jtype
		for i = 1; i <= n; i++ {
			for j = 1; j <= n; j++ {
				a.Set(i-1, j-1, alist[_i][(i-1)*(n)+j-1])
			}
		}
		for i = 1; i <= n; i++ {
			wr = wlist[_i][(i-1)*(4)+0]
			wi = wlist[_i][(i-1)*(4)+1]
			rcdein.Set(i-1, wlist[_i][(i-1)*(4)+2])
			rcdvin.Set(i-1, wlist[_i][(i-1)*(4)+3])
			w1.Set(i-1, complex(wr, wi))
		}
		err = zget23(true, isrt, 'N', 22, thresh, iseed, n, a, h, w, w1, vl, vr, lre, rcondv, rcndv1, rcdvin, rconde, rcnde1, rcdein, scale, scale1, result, work, 6*n+2*pow(n, 2), rwork)

		//     Check for RESULT(j) > THRESH
		ntest = 0
		nfail = 0
		for j = 1; j <= 11; j++ {
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
			fmt.Printf("\n %3s -- Complex Eigenvalue-Eigenvector Decomposition Expert Driver\n Matrix types (see zdrvvx for details): \n", path)
			fmt.Printf("\n Special Matrices:\n  1=Zero matrix.                          5=Diagonal: geometr. spaced entries.\n  2=Identity matrix.                      6=Diagonal: clustered entries.\n  3=Transposed Jordan block.              7=Diagonal: large, evenly spaced.\n  4=Diagonal: evenly spaced entries.      8=Diagonal: small, evenly spaced.\n")
			fmt.Printf(" Dense, Non-Symmetric Matrices:\n  9=Well-cond., evenly spaced eigenvals. 14=Ill-cond., geomet. spaced eigenals.\n 10=Well-cond., geom. spaced eigenvals.  15=Ill-conditioned, clustered e.vals.\n 11=Well-conditioned, clustered e.vals.  16=Ill-cond., random complex \n 12=Well-cond., random complex           17=Ill-cond., large rand. complx \n 13=Ill-conditioned, evenly spaced.      18=Ill-cond., small rand. complx \n")
			fmt.Printf(" 19=Matrix with random O(1) entries.     21=Matrix with small random entries.\n 20=Matrix with large random entries.    22=Matrix read from input file\n\n")
			fmt.Printf(" Tests performed with test threshold =%8.2f\n\n 1 = | A VR - VR W | / ( n |A| ulp ) \n 2 = | transpose(A) VL - VL W | / ( n |A| ulp ) \n 3 = | |VR(i)| - 1 | / ulp \n 4 = | |VL(i)| - 1 | / ulp \n 5 = 0 if W same no matter if VR or VL computed, 1/ulp otherwise\n 6 = 0 if VR same no matter what else computed,  1/ulp otherwise\n 7 = 0 if VL same no matter what else computed,  1/ulp otherwise\n 8 = 0 if RCONDV same no matter what else computed,  1/ulp otherwise\n 9 = 0 if SCALE, ILO, IHI, ABNRM same no matter what else computed,  1/ulp otherwise\n 10 = | RCONDV - RCONDV(precomputed) | / cond(RCONDV),\n 11 = | RCONDE - RCONDE(precomputed) | / cond(RCONDE),\n", thresh)
			ntestf = 2
		}

		for j = 1; j <= 11; j++ {
			if result.Get(j-1) >= thresh {
				fmt.Printf(" N=%5d, input example =%3d,  test(%2d)=%10.3f\n", n, jtype, j, result.Get(j-1))
				err = fmt.Errorf(" N=%5d, input example =%3d,  test(%2d)=%10.3f\n", n, jtype, j, result.Get(j-1))
			}
		}

		nerrs = nerrs + nfail
		ntestt = ntestt + ntest
	}

	//     Summary
	dlasum(path, nerrs, ntestt)

	return
}
