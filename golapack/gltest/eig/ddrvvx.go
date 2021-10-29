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

// ddrvvx checks the nonsymmetric eigenvalue problem expert driver
//    DGEEVX.
//
//    ddrvvx uses both test matrices generated randomly depending on
//    data supplied in the calling sequence, as well as on data
//    read from an input file and including precomputed condition
//    numbers to which it compares the ones it computes.
//
//    When ddrvvx is called, a number of matrix "sizes" ("n's") and a
//    number of matrix "types" are specified in the calling sequence.
//    For each size ("n") and each type of matrix, one matrix will be
//    generated and used to test the nonsymmetric eigenroutines.  For
//    each matrix, 9 tests will be performed:
//
//    (1)     | A * VR - VR * W | / ( n |A| ulp )
//
//      Here VR is the matrix of unit right eigenvectors.
//      W is a block diagonal matrix, with a 1x1 block for each
//      real eigenvalue and a 2x2 block for each complex conjugate
//      pair.  If eigenvalues j and j+1 are a complex conjugate pair,
//      so WR(j) = WR(j+1) = wr and WI(j) = - WI(j+1) = wi, then the
//      2 x 2 block corresponding to the pair will be:
//
//              (  wr  wi  )
//              ( -wi  wr  )
//
//      Such a block multiplying an n x 2 matrix  ( ur ui ) on the
//      right will be the same as multiplying  ur + i*ui  by  wr + i*wi.
//
//    (2)     | A**H * VL - VL * W**H | / ( n |A| ulp )
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
//    eigenvalues and reciprocal condition numbers for the eigenvalues
//    and right eigenvectors. For these matrices, in addition to tests
//    (1) to (9) we will compute the following two tests:
//
//   (10)  |RCONDV - RCDVIN| / cond(RCONDV)
//
//      RCONDV is the reciprocal right eigenvector condition number
//      computed by DGEEVX and RCDVIN (the precomputed true value)
//      is supplied as input. cond(RCONDV) is the condition number of
//      RCONDV, and takes errors in computing RCONDV into account, so
//      that the resulting quantity should be O(ULP). cond(RCONDV) is
//      essentially given by norm(A)/RCONDE.
//
//   (11)  |RCONDE - RCDEIN| / cond(RCONDE)
//
//      RCONDE is the reciprocal eigenvalue condition number
//      computed by DGEEVX and RCDEIN (the precomputed true value)
//      is supplied as input.  cond(RCONDE) is the condition number
//      of RCONDE, and takes errors in computing RCONDE into account,
//      so that the resulting quantity should be O(ULP). cond(RCONDE)
//      is essentially given by norm(A)/RCONDV.
func ddrvvx(nsizes int, nn []int, ntypes int, dotype []bool, iseed []int, thresh float64, nounit int, a, h *mat.Matrix, wr, wi, wr1, wi1 *mat.Vector, vl, vr, lre *mat.Matrix, rcondv, rcndv1, rcdvin, rconde, rcnde1, rcdein, scale, scale1, result, work *mat.Vector, nwork int, iwork []int, t *testing.T) (err error) {
	var badnn bool
	var balanc byte
	var anorm, cond, conds, one, ovfl, rtulp, rtulpi, ulp, ulpinv, unfl, zero float64
	var _i, i, ibal, iinfo, imode, itype, iwk, j, jcol, jsize, jtype, maxtyp, mtypes, n, nerrs, nfail, nmax, nnwork, ntest, ntestf, ntestt int
	adumma := make([]byte, 1)
	bal := make([]byte, 4)
	idumma := make([]int, 1)
	ioldsd := make([]int, 4)
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
	bal[0], bal[1], bal[2], bal[3] = 'N', 'P', 'S', 'B'

	path := "Dvx"

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
		gltest.Xerbla2("ddrvvx", err)
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
				fmt.Printf(" ddrvvx: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Generator", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				return
			}

		label90:
			;

			//           Test for minimal and generous workspace
			for iwk = 1; iwk <= 3; iwk++ {
				if iwk == 1 {
					nnwork = 3 * n
				} else if iwk == 2 {
					nnwork = 6*n + pow(n, 2)
				} else {
					nnwork = 6*n + 2*pow(n, 2)
				}
				nnwork = max(nnwork, 1)

				//              Test for all balancing options
				for ibal = 1; ibal <= 4; ibal++ {
					balanc = bal[ibal-1]

					//                 Perform tests
					iinfo, err = dget23(false, balanc, jtype, thresh, ioldsd, nounit, n, a, h, wr, wi, wr1, wi1, vl, vr, lre, rcondv, rcndv1, rcdvin, rconde, rcnde1, rcdein, scale, scale1, result, work, nnwork, &iwork)

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
						t.Fail()
						ntestf = ntestf + 1
					}
					if ntestf == 1 {
						fmt.Printf("\n %3s -- Real Eigenvalue-Eigenvector Decomposition Expert Driver\n Matrix types (see ddrvvx for details): \n", path)
						fmt.Printf("\n Special Matrices:\n  1=Zero matrix.                          5=Diagonal: geometr. spaced entries.\n  2=Identity matrix.                      6=Diagonal: clustered entries.\n  3=Transposed Jordan block.              7=Diagonal: large, evenly spaced.\n  4=Diagonal: evenly spaced entries.      8=Diagonal: small, evenly spaced.\n")
						fmt.Printf(" Dense, Non-Symmetric Matrices:\n  9=Well-cond., evenly spaced eigenvals. 14=Ill-cond., geomet. spaced eigenals.\n 10=Well-cond., geom. spaced eigenvals.  15=Ill-conditioned, clustered e.vals.\n 11=Well-conditioned, clustered e.vals.  16=Ill-cond., random complex \n 12=Well-cond., random complex           17=Ill-cond., large rand. complx \n 13=Ill-conditioned, evenly spaced.      18=Ill-cond., small rand. complx \n")
						fmt.Printf(" 19=Matrix with random O(1) entries.     21=Matrix with small random entries.\n 20=Matrix with large random entries.    22=Matrix read from input file\n\n")
						fmt.Printf(" Tests performed with test threshold =%8.2f\n\n 1 = | A VR - VR W | / ( n |A| ulp ) \n 2 = | transpose(A) VL - VL W | / ( n |A| ulp ) \n 3 = | |VR(i)| - 1 | / ulp \n 4 = | |VL(i)| - 1 | / ulp \n 5 = 0 if W same no matter if VR or VL computed, 1/ulp otherwise\n 6 = 0 if VR same no matter what else computed,  1/ulp otherwise\n 7 = 0 if VL same no matter what else computed,  1/ulp otherwise\n 8 = 0 if RCONDV same no matter what else computed,  1/ulp otherwise\n 9 = 0 if SCALE, ILO, IHI, ABNRM same no matter what else computed,  1/ulp otherwise\n 10 = | RCONDV - RCONDV(precomputed) | / cond(RCONDV),\n 11 = | RCONDE - RCONDE(precomputed) | / cond(RCONDE),\n", thresh)
						ntestf = 2
					}

					for j = 1; j <= 9; j++ {
						if result.Get(j-1) >= thresh {
							t.Fail()
							fmt.Printf(" BALANC='%c',N=%4d,IWK=%1d, seed=%4d, type %2d, test(%2d)=%10.3f\n", balanc, n, iwk, ioldsd, jtype, j, result.Get(j-1))
						}
					}

					nerrs = nerrs + nfail
					ntestt = ntestt + ntest

				}
			}
		label140:
		}
	}

	nlist := []int{1, 1, 2, 2, 2, 6, 4, 5, 5, 6, 6, 6, 4, 6, 5, 10, 4, 6, 10, 4, 6, 4, 3, 6, 6, 6, 6, 6, 12, 6, 6, 6, 6, 8, 6, 4, 5, 6, 10}
	alist := [][]float64{
		{
			0.0000e+00,
		},
		{
			1.0000e+00,
		},
		{
			0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00,
		},
		{
			3.0000e+00, 2.0000e+00,
			2.0000e+00, 3.0000e+00,
		},
		{
			3.0000e+00, -2.0000e+00,
			2.0000e+00, 3.0000e+00,
		},
		{
			1.0000e-07, -1.0000e-07, 1.0000e+00, 1.1000e+00, 2.3000e+00, 3.7000e+00,
			3.0000e-07, 1.0000e-07, 1.0000e+00, 1.0000e+00, -1.3000e+00, -7.7000e+00,
			0.0000e+00, 0.0000e+00, 3.0000e-07, 1.0000e-07, 2.2000e+00, 3.3000e+00,
			0.0000e+00, 0.0000e+00, -1.0000e-07, 3.0000e-07, 1.8000e+00, 1.6000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 4.0000e-06, 5.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 3.0000e+00, 4.0000e-06,
		},
		{
			7.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
			-1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
			-1.0000e+00, 1.0000e+00, 5.0000e+00, -3.0000e+00,
			1.0000e+00, -1.0000e+00, 3.0000e+00, 3.0000e+00,
		},
		{
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
		},
		{
			1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00,
		},
		{
			1.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 1.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00,
		},
		{
			1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			1.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 1.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00,
		},
		{
			1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 2.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 3.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 4.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 5.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 6.0000e+00,
		},
		{
			9.4480e-01, 6.7670e-01, 6.9080e-01, 5.9650e-01,
			5.8760e-01, 8.6420e-01, 6.7690e-01, 7.2600e-02,
			7.2560e-01, 1.9430e-01, 9.6870e-01, 2.8310e-01,
			2.8490e-01, 5.8000e-02, 4.8450e-01, 7.3610e-01,
		},
		{
			5.0410e-01, 6.6520e-01, 7.7190e-01, 6.3870e-01, 5.9550e-01, 6.1310e-01,
			1.5740e-01, 3.7340e-01, 5.9840e-01, 1.5470e-01, 9.4270e-01, 6.5900e-02,
			4.4170e-01, 7.2300e-02, 1.5440e-01, 5.4920e-01, 8.7000e-03, 3.0040e-01,
			2.0080e-01, 6.0800e-01, 3.0340e-01, 8.4390e-01, 2.3900e-01, 5.7680e-01,
			9.3610e-01, 7.4130e-01, 1.4440e-01, 1.7860e-01, 1.4280e-01, 7.2630e-01,
			5.5990e-01, 9.3360e-01, 7.8000e-02, 4.0930e-01, 6.7140e-01, 5.6170e-01,
		},
		{
			2.0000e-03, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 1.0000e-03, 1.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, -1.0000e-03, 1.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, -2.0000e-03, 1.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
		},
		{
			4.8630e-01, 9.1260e-01, 2.1900e-02, 6.0110e-01, 1.4050e-01, 2.0840e-01, 8.2640e-01, 8.4410e-01, 3.1420e-01, 8.6750e-01,
			7.1500e-01, 2.6480e-01, 8.8510e-01, 2.6150e-01, 5.9520e-01, 4.7800e-01, 7.6730e-01, 4.6110e-01, 5.7320e-01, 7.7000e-03,
			2.1210e-01, 5.5080e-01, 5.2350e-01, 3.0810e-01, 6.6020e-01, 2.8900e-01, 2.3140e-01, 2.2790e-01, 9.6600e-02, 1.0910e-01,
			7.1510e-01, 8.5790e-01, 5.7710e-01, 5.1140e-01, 1.9010e-01, 9.0810e-01, 6.0090e-01, 7.1980e-01, 1.0640e-01, 8.6840e-01,
			5.6800e-01, 2.8100e-02, 4.0140e-01, 6.3150e-01, 1.1480e-01, 7.5800e-02, 9.4230e-01, 7.2030e-01, 3.6850e-01, 1.7430e-01,
			7.7210e-01, 3.0280e-01, 5.5640e-01, 9.9980e-01, 3.6520e-01, 5.2580e-01, 3.7030e-01, 6.7790e-01, 9.9350e-01, 5.0270e-01,
			7.3960e-01, 4.5600e-02, 7.4740e-01, 9.2880e-01, 2.2000e-03, 8.2600e-02, 3.6340e-01, 4.9120e-01, 9.4050e-01, 3.8910e-01,
			5.6370e-01, 8.5540e-01, 3.2100e-02, 2.6380e-01, 3.6090e-01, 6.4970e-01, 8.4690e-01, 9.3500e-01, 3.7000e-02, 2.9170e-01,
			8.6560e-01, 6.3270e-01, 3.5620e-01, 6.3560e-01, 2.7360e-01, 6.5120e-01, 1.0220e-01, 2.8880e-01, 5.7620e-01, 4.0790e-01,
			5.3320e-01, 4.1210e-01, 7.2870e-01, 2.3110e-01, 6.8300e-01, 7.3860e-01, 8.1800e-01, 9.8150e-01, 8.0550e-01, 2.5660e-01,
		},
		{
			-3.8730e-01, 3.6560e-01, 3.1200e-02, -5.8340e-01,
			5.5230e-01, -1.1854e+00, 9.8330e-01, 7.6670e-01,
			1.6746e+00, -1.9900e-02, -1.8293e+00, 5.7180e-01,
			-5.2500e-01, 3.5340e-01, -2.7210e-01, -8.8300e-02,
		},
		{
			-1.0777e+00, 1.7027e+00, 2.6510e-01, 8.5160e-01, 1.0121e+00, 2.5710e-01,
			-1.3400e-02, 3.9030e-01, -1.2680e+00, 2.7530e-01, -3.2350e-01, -1.3844e+00,
			1.5230e-01, 3.0680e-01, 8.7330e-01, -3.3410e-01, -4.8310e-01, -1.5416e+00,
			1.4470e-01, -6.0570e-01, 3.1900e-02, -1.0905e+00, -8.3700e-02, 6.2410e-01,
			-7.6510e-01, -1.7889e+00, -1.5069e+00, -6.0210e-01, 5.2170e-01, 6.4700e-01,
			8.1940e-01, 2.1100e-01, 5.4320e-01, 7.5610e-01, 1.7130e-01, 5.5400e-01,
		},
		{
			-1.0639e+00, 1.6120e-01, 1.5620e-01, 3.4360e-01, -6.7480e-01, 1.6598e+00, 6.4650e-01, -7.8630e-01, -2.6100e-01, 7.0190e-01,
			-8.4400e-01, -2.2439e+00, 1.8800e+00, -1.0005e+00, 7.4500e-02, -1.6156e+00, 2.8220e-01, 8.5600e-01, 1.3497e+00, -1.5883e+00,
			1.5988e+00, 1.1758e+00, 1.2398e+00, 1.1173e+00, 2.1500e-01, 4.3140e-01, 1.8500e-01, 7.9470e-01, 6.6260e-01, 8.6460e-01,
			-2.2960e-01, 1.2442e+00, 2.3242e+00, -5.0690e-01, -7.5160e-01, -5.4370e-01, -2.5990e-01, 1.2830e+00, -1.1067e+00, -1.1150e-01,
			-3.6040e-01, 4.0420e-01, 6.1240e-01, -1.2164e+00, -9.4650e-01, -3.1460e-01, 1.8310e-01, 7.3710e-01, 1.4278e+00, 2.9220e-01,
			4.6150e-01, 3.8740e-01, -4.2900e-02, -9.3600e-01, 7.1160e-01, -8.2590e-01, -1.7640e+00, -9.4660e-01, 1.8202e+00, -2.5480e-01,
			1.2934e+00, -9.7550e-01, 6.7480e-01, -1.0481e+00, -1.8442e+00, -5.4600e-02, 7.4050e-01, 6.1000e-03, 1.2430e+00, -1.8490e-01,
			-3.4710e-01, -9.5800e-01, 1.6530e-01, 9.1300e-02, -5.2010e-01, -1.1832e+00, 8.5410e-01, -2.3200e-01, -1.6155e+00, 5.5180e-01,
			1.0190e+00, -6.8240e-01, 8.0850e-01, 2.5950e-01, -3.7580e-01, -1.8825e+00, 1.6473e+00, -6.5920e-01, 8.0250e-01, -4.9000e-03,
			1.2670e+00, -4.2400e-02, 8.9570e-01, -1.6770e-01, 1.4620e-01, 9.8800e-01, -2.3170e-01, -1.4483e+00, -5.8200e-02, 1.9700e-02,
		},
		{
			-1.2298e+00, -2.3142e+00, -6.9800e-02, 1.0523e+00,
			2.0390e-01, -1.2298e+00, 8.0500e-02, 9.7860e-01,
			0.0000e+00, 0.0000e+00, 2.5600e-01, -8.9100e-01,
			0.0000e+00, 0.0000e+00, 2.7480e-01, 2.5600e-01,
		},
		{
			5.9930e-01, 1.9372e+00, -1.6160e-01, -1.4602e+00, 6.0180e-01, 2.7120e+00,
			-2.2049e+00, 5.9930e-01, -1.0679e+00, 1.9405e+00, -1.4400e+00, -2.2110e-01,
			0.0000e+00, 0.0000e+00, -2.4567e+00, -6.8650e-01, -1.9101e+00, 6.4960e-01,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 7.3620e-01, 3.9700e-01, -1.5190e-01,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, -1.0034e+00, 1.1954e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, -1.3400e-01, -1.0034e+00,
		},
		{
			1.0000e-04, 1.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, -1.0000e-04, 1.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 1.0000e-02, 1.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, -5.0000e-03,
		},
		{
			2.0000e-06, 1.0000e+00, -2.0000e+00,
			1.0000e-06, -2.0000e+00, 4.0000e+00,
			0.0000e+00, 1.0000e+00, -2.0000e+00,
		},
		{
			2.4080e-01, 6.5530e-01, 9.1660e-01, 5.0300e-02, 2.8490e-01, 2.4080e-01,
			6.9070e-01, 9.7000e-01, 1.4020e-01, 5.7820e-01, 6.7670e-01, 6.9070e-01,
			1.0620e-01, 3.8000e-02, 7.0540e-01, 2.4320e-01, 8.6420e-01, 1.0620e-01,
			2.6400e-01, 9.8800e-02, 1.7800e-02, 9.4480e-01, 1.9430e-01, 2.6400e-01,
			7.0340e-01, 2.5600e-01, 2.6110e-01, 5.8760e-01, 5.8000e-02, 7.0340e-01,
			4.0210e-01, 5.5980e-01, 1.3580e-01, 7.2560e-01, 6.9080e-01, 4.0210e-01,
		},
		{
			3.4800e+00, -2.9900e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			-4.9000e-01, 2.4800e+00, -1.9900e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, -4.9000e-01, 1.4800e+00, -9.9000e-01, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, -9.9000e-01, 1.4800e+00, -4.9000e-01, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, -1.9900e+00, 2.4800e+00, -4.9000e-01,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, -2.9900e+00, 3.4800e+00,
		},
		{
			0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00,
			1.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 1.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 1.0000e+00,
			-1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
		},
		{
			3.5345e-01, 9.3023e-01, 7.4679e-02, -1.0059e-02, 4.6698e-02, -4.3480e-02,
			9.3545e-01, -3.5147e-01, -2.8216e-02, 3.8008e-03, -1.7644e-02, 1.6428e-02,
			0.0000e+00, -1.0555e-01, 7.5211e-01, -1.0131e-01, 4.7030e-01, -4.3789e-01,
			0.0000e+00, 0.0000e+00, 6.5419e-01, 1.1779e-01, -5.4678e-01, 5.0911e-01,
			0.0000e+00, 0.0000e+00, 0.0000e+00, -9.8780e-01, -1.1398e-01, 1.0612e-01,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 6.8144e-01, 7.3187e-01,
		},
		{
			1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
			5.0000e-01, 3.3330e-01, 2.5000e-01, 2.0000e-01, 1.6670e-01, 1.4290e-01,
			3.3330e-01, 2.5000e-01, 2.0000e-01, 1.6670e-01, 1.4290e-01, 1.2500e-01,
			2.5000e-01, 2.0000e-01, 1.6670e-01, 1.4290e-01, 1.2500e-01, 1.1110e-01,
			2.0000e-01, 1.6670e-01, 1.4290e-01, 1.2500e-01, 1.1110e-01, 1.0000e-01,
			1.6670e-01, 1.4290e-01, 1.2500e-01, 1.1110e-01, 1.0000e-01, 9.0900e-02,
		},
		{
			1.2000e+01, 1.1000e+01, 1.0000e+01, 9.0000e+00, 8.0000e+00, 7.0000e+00, 6.0000e+00, 5.0000e+00, 4.0000e+00, 3.0000e+00, 2.0000e+00, 1.0000e+00,
			1.1000e+01, 1.1000e+01, 1.0000e+01, 9.0000e+00, 8.0000e+00, 7.0000e+00, 6.0000e+00, 5.0000e+00, 4.0000e+00, 3.0000e+00, 2.0000e+00, 1.0000e+00,
			0.0000e+00, 1.0000e+01, 1.0000e+01, 9.0000e+00, 8.0000e+00, 7.0000e+00, 6.0000e+00, 5.0000e+00, 4.0000e+00, 3.0000e+00, 2.0000e+00, 1.0000e+00,
			0.0000e+00, 0.0000e+00, 9.0000e+00, 9.0000e+00, 8.0000e+00, 7.0000e+00, 6.0000e+00, 5.0000e+00, 4.0000e+00, 3.0000e+00, 2.0000e+00, 1.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 8.0000e+00, 8.0000e+00, 7.0000e+00, 6.0000e+00, 5.0000e+00, 4.0000e+00, 3.0000e+00, 2.0000e+00, 1.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 7.0000e+00, 7.0000e+00, 6.0000e+00, 5.0000e+00, 4.0000e+00, 3.0000e+00, 2.0000e+00, 1.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 6.0000e+00, 6.0000e+00, 5.0000e+00, 4.0000e+00, 3.0000e+00, 2.0000e+00, 1.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 5.0000e+00, 5.0000e+00, 4.0000e+00, 3.0000e+00, 2.0000e+00, 1.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 4.0000e+00, 4.0000e+00, 3.0000e+00, 2.0000e+00, 1.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 3.0000e+00, 3.0000e+00, 2.0000e+00, 1.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.0000e+00, 2.0000e+00, 1.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00,
		},
		{
			0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			5.0000e+00, 0.0000e+00, 2.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 4.0000e+00, 0.0000e+00, 3.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 3.0000e+00, 0.0000e+00, 4.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 2.0000e+00, 0.0000e+00, 5.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
		},
		{
			1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00,
			-1.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00,
			-1.0000e+00, -1.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00,
			-1.0000e+00, -1.0000e+00, -1.0000e+00, 1.0000e+00, 0.0000e+00, 1.0000e+00,
			-1.0000e+00, -1.0000e+00, -1.0000e+00, -1.0000e+00, 1.0000e+00, 1.0000e+00,
			-1.0000e+00, -1.0000e+00, -1.0000e+00, -1.0000e+00, -1.0000e+00, 1.0000e+00,
		},
		{
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00,
			1.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00,
			0.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 0.0000e+00, 1.0000e+00,
			1.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
			1.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 0.0000e+00,
			0.0000e+00, 1.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
		},
		{
			1.0000e+00, 4.0112e+00, 1.2750e+01, 4.0213e+01, 1.2656e+02, 3.9788e+02,
			1.0000e+00, 3.2616e+00, 1.0629e+01, 3.3342e+01, 1.0479e+02, 3.2936e+02,
			1.0000e+00, 3.1500e+00, 9.8006e+00, 3.0630e+01, 9.6164e+01, 3.0215e+02,
			1.0000e+00, 3.2755e+00, 1.0420e+01, 3.2957e+01, 1.0374e+02, 3.2616e+02,
			1.0000e+00, 2.8214e+00, 8.4558e+00, 2.6296e+01, 8.2443e+01, 2.5893e+02,
			1.0000e+00, 2.6406e+00, 8.3565e+00, 2.6558e+01, 8.3558e+01, 2.6268e+02,
		},
		{
			0.0000e+00, 4.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			1.0000e+00, 0.0000e+00, 4.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 1.0000e+00, 0.0000e+00, 4.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 4.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 4.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 4.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 4.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
		},
		{
			8.5000e+00, -1.0472e+01, 2.8944e+00, -1.5279e+00, 1.1056e+00, -5.0000e-01,
			2.6180e+00, -1.1708e+00, -2.0000e+00, 8.9440e-01, -6.1800e-01, 2.7640e-01,
			-7.2360e-01, 2.0000e+00, -1.7080e-01, -1.6180e+00, 8.9440e-01, -3.8200e-01,
			3.8200e-01, -8.9440e-01, 1.6180e+00, 1.7080e-01, -2.0000e+00, 7.2360e-01,
			-2.7640e-01, 6.1800e-01, -8.9440e-01, 2.0000e+00, 1.1708e+00, -2.6180e+00,
			5.0000e-01, -1.1056e+00, 1.5279e+00, -2.8944e+00, 1.0472e+01, -8.5000e+00,
		},
		{
			4.0000e+00, -5.0000e+00, 0.0000e+00, 3.0000e+00,
			0.0000e+00, 4.0000e+00, -3.0000e+00, -5.0000e+00,
			5.0000e+00, -3.0000e+00, 4.0000e+00, 0.0000e+00,
			3.0000e+00, 0.0000e+00, 5.0000e+00, 4.0000e+00,
		},
		{
			1.5000e+01, 1.1000e+01, 6.0000e+00, -9.0000e+00, -1.5000e+01,
			1.0000e+00, 3.0000e+00, 9.0000e+00, -3.0000e+00, -8.0000e+00,
			7.0000e+00, 6.0000e+00, 6.0000e+00, -3.0000e+00, -1.1000e+01,
			7.0000e+00, 7.0000e+00, 5.0000e+00, -3.0000e+00, -1.1000e+01,
			1.7000e+01, 1.2000e+01, 5.0000e+00, -1.0000e+01, -1.6000e+01,
		},
		{
			-9.0000e+00, 2.1000e+01, -1.5000e+01, 4.0000e+00, 2.0000e+00, 0.0000e+00,
			-1.0000e+01, 2.1000e+01, -1.4000e+01, 4.0000e+00, 2.0000e+00, 0.0000e+00,
			-8.0000e+00, 1.6000e+01, -1.1000e+01, 4.0000e+00, 2.0000e+00, 0.0000e+00,
			-6.0000e+00, 1.2000e+01, -9.0000e+00, 3.0000e+00, 3.0000e+00, 0.0000e+00,
			-4.0000e+00, 8.0000e+00, -6.0000e+00, 0.0000e+00, 5.0000e+00, 0.0000e+00,
			-2.0000e+00, 4.0000e+00, -3.0000e+00, 0.0000e+00, 1.0000e+00, 3.0000e+00,
		},
		{
			1.0000e+00, 1.0000e+00, 1.0000e+00, -2.0000e+00, 1.0000e+00, -1.0000e+00, 2.0000e+00, -2.0000e+00, 4.0000e+00, -3.0000e+00,
			-1.0000e+00, 2.0000e+00, 3.0000e+00, -4.0000e+00, 2.0000e+00, -2.0000e+00, 4.0000e+00, -4.0000e+00, 8.0000e+00, -6.0000e+00,
			-1.0000e+00, 0.0000e+00, 5.0000e+00, -5.0000e+00, 3.0000e+00, -3.0000e+00, 6.0000e+00, -6.0000e+00, 1.2000e+01, -9.0000e+00,
			-1.0000e+00, 0.0000e+00, 3.0000e+00, -4.0000e+00, 4.0000e+00, -4.0000e+00, 8.0000e+00, -8.0000e+00, 1.6000e+01, -1.2000e+01,
			-1.0000e+00, 0.0000e+00, 3.0000e+00, -6.0000e+00, 5.0000e+00, -4.0000e+00, 1.0000e+01, -1.0000e+01, 2.0000e+01, -1.5000e+01,
			-1.0000e+00, 0.0000e+00, 3.0000e+00, -6.0000e+00, 2.0000e+00, -2.0000e+00, 1.2000e+01, -1.2000e+01, 2.4000e+01, -1.8000e+01,
			-1.0000e+00, 0.0000e+00, 3.0000e+00, -6.0000e+00, 2.0000e+00, -5.0000e+00, 1.5000e+01, -1.3000e+01, 2.8000e+01, -2.1000e+01,
			-1.0000e+00, 0.0000e+00, 3.0000e+00, -6.0000e+00, 2.0000e+00, -5.0000e+00, 1.2000e+01, -1.1000e+01, 3.2000e+01, -2.4000e+01,
			-1.0000e+00, 0.0000e+00, 3.0000e+00, -6.0000e+00, 2.0000e+00, -5.0000e+00, 1.2000e+01, -1.4000e+01, 3.7000e+01, -2.6000e+01,
			-1.0000e+00, 0.0000e+00, 3.0000e+00, -6.0000e+00, 2.0000e+00, -5.0000e+00, 1.2000e+01, -1.4000e+01, 3.6000e+01, -2.5000e+01,
		},
	}
	wr1list := [][]float64{
		{0.0000e+00},
		{1.0000e+00},
		{0.0000e+00, 0.0000e+00},
		{1.0000e+00, 5.0000e+00},
		{3.0000e+00, 3.0000e+00},
		{-3.8730e+00, 1.0000e-07, 1.0000e-07, 3.0000e-07, 3.0000e-07, 3.8730e+00},
		{3.9603e+00, 3.9603e+00, 4.0397e+00, 4.0397e+00},
		{0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00},
		{1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00},
		{1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00},
		{1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00},
		{1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00, 5.0000e+00, 6.0000e+00},
		{2.4326e-01, 2.4326e-01, 7.4091e-01, 2.2864e+00},
		{-5.2278e-01, -3.5380e-01, -8.0876e-03, 3.4760e-01, 3.4760e-01, 2.7698e+00},
		{-2.0000e-03, -1.0000e-03, 0.0000e+00, 1.0000e-03, 2.0000e-03},
		{-4.6121e-01, -4.6121e-01, -4.5164e-01, -1.4922e-01, -1.4922e-01, 3.3062e-02, 3.0849e-01, 3.0849e-01, 5.4509e-01, 5.0352e+00},
		{-1.8952e+00, -1.8952e+00, -9.5162e-02, 3.9520e-01},
		{-1.7029e+00, -1.0307e+00, 2.8487e-01, 2.8487e-01, 1.1675e+00, 1.1675e+00},
		{-2.6992e+00, -2.6992e+00, -2.4366e+00, -1.2882e+00, -1.2882e+00, 9.0275e-01, 9.0442e-01, 9.0442e-01, 1.6774e+00, 3.0060e+00},
		{-1.2298e+00, -1.2298e+00, 2.5600e-01, 2.5600e-01},
		{-2.4567e+00, -1.0034e+00, -1.0034e+00, 5.9930e-01, 5.9930e-01, 7.3620e-01},
		{-5.0000e-03, -1.0000e-04, 1.0000e-04, 1.0000e-02},
		{-4.0000e+00, 0.0000e+00, 2.2096e-06},
		{-3.4008e-01, -3.4008e-01, -1.6998e-07, 7.2311e-01, 7.2311e-01, 2.5551e+00},
		{1.3034e-02, 1.1294e+00, 2.0644e+00, 2.8388e+00, 4.3726e+00, 4.4618e+00},
		{-1.7321e+00, -1.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 1.7321e+00},
		{-9.9980e-01, -9.9980e-01, 7.4539e-01, 7.4539e-01, 9.9929e-01, 9.9929e-01},
		{-2.2135e-01, -3.1956e-02, -8.5031e-04, -5.8584e-05, 1.3895e-05, 2.1324e+00},
		{-2.8234e-02, 7.2587e-02, 7.2587e-02, 1.8533e-01, 2.8828e-01, 6.4315e-01, 1.5539e+00, 3.5119e+00, 6.9615e+00, 1.2311e+01, 2.0199e+01, 3.2229e+01},
		{-5.0000e+00, -3.0000e+00, -1.0000e+00, 1.0000e+00, 3.0000e+00, 5.0000e+00},
		{8.0298e-02, 8.0298e-02, 1.4415e+00, 1.4415e+00, 1.4782e+00, 1.4782e+00},
		{-3.5343e-02, -3.5343e-02, 5.8440e-07, 6.4087e-01, 6.4087e-01, 3.7889e+00},
		{-5.3220e-01, -1.0118e-01, -9.8749e-03, 2.9861e-03, 1.8075e-01, 3.9260e+02},
		{-3.7588e+00, -3.0642e+00, -2.0000e+00, -6.9459e-01, 6.9459e-01, 2.0000e+00, 3.0642e+00, 3.7588e+00},
		{-5.8930e-01, -2.7627e-01, -2.7627e-01, 2.7509e-01, 2.7509e-01, 5.9167e-01},
		{1.0000e+00, 1.0000e+00, 2.0000e+00, 1.2000e+01},
		{-9.9999e-01, 1.4980e+00, 1.4980e+00, 1.5020e+00, 1.5020e+00},
		{1.0000e+00, 1.0000e+00, 2.0000e+00, 2.0000e+00, 3.0000e+00, 3.0000e+00},
		{1.0000e+00, 1.9867e+00, 2.0000e+00, 2.0000e+00, 2.0067e+00, 2.0067e+00, 2.9970e+00, 3.0000e+00, 3.0000e+00, 3.0030e+00},
	}
	wi1list := [][]float64{
		{0.0000e+00},
		{0.0000e+00},
		{0.0000e+00, 0.0000e+00},
		{0.0000e+00, 0.0000e+00},
		{2.0000e+00, -2.0000e+00},
		{0.0000e+00, 1.7321e-07, -1.7321e-07, 1.0000e-07, -1.0000e-07, 0.0000e+00},
		{4.0425e-02, -4.0425e-02, 3.8854e-02, -3.8854e-02},
		{0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00},
		{0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00},
		{0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00},
		{0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00},
		{0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00},
		{2.1409e-01, -2.1409e-01, 0.0000e+00, 0.0000e+00},
		{0.0000e+00, 0.0000e+00, 0.0000e+00, 3.0525e-01, -3.0525e-01, 0.0000e+00},
		{0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00},
		{7.2657e-01, -7.2657e-01, 0.0000e+00, 4.8255e-01, -4.8255e-01, 0.0000e+00, 1.1953e-01, -1.1953e-01, 0.0000e+00, 0.0000e+00},
		{7.5059e-01, -7.5059e-01, 0.0000e+00, 0.0000e+00},
		{0.0000e+00, 0.0000e+00, 1.2101e+00, -1.2101e+00, 4.6631e-01, -4.6631e-01},
		{9.0387e-01, -9.0387e-01, 0.0000e+00, 8.8930e-01, -8.8930e-01, 0.0000e+00, 2.5661e+00, -2.5661e+00, 0.0000e+00, 0.0000e+00},
		{6.8692e-01, -6.8692e-01, 4.9482e-01, -4.9482e-01},
		{0.0000e+00, 4.0023e-01, -4.0023e-01, 2.0667e+00, -2.0667e+00, 0.0000e+00},
		{0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00},
		{0.0000e+00, 0.0000e+00, 0.0000e+00},
		{3.2133e-01, -3.2133e-01, 0.0000e+00, 5.9389e-02, -5.9389e-02, 0.0000e+00},
		{0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00},
		{0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00},
		{1.9645e-02, -1.9645e-02, 6.6663e-01, -6.6663e-01, 3.7545e-02, -3.7545e-02},
		{0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00},
		{0.0000e+00, 9.0746e-02, -9.0746e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00},
		{0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00},
		{2.4187e+00, -2.4187e+00, 6.2850e-01, -6.2850e-01, 1.5638e-01, -1.5638e-01},
		{7.4812e-01, -7.4812e-01, 0.0000e+00, 7.2822e-01, -7.2822e-01, 0.0000e+00},
		{0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00},
		{0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00},
		{0.0000e+00, 4.9852e-01, -4.9852e-01, 5.0059e-01, -5.0059e-01, 0.0000e+00},
		{5.0000e+00, -5.0000e+00, 0.0000e+00, 0.0000e+00},
		{0.0000e+00, 3.5752e+00, -3.5752e+00, 3.5662e+00, -3.5662e+00},
		{6.2559e-04, -6.2559e-04, 1.0001e+00, -1.0001e+00, 0.0000e+00, 0.0000e+00},
		{0.0000e+00, 0.0000e+00, 2.5052e-03, -2.5052e-03, 1.1763e-02, -1.1763e-02, 0.0000e+00, 8.7028e-04, -8.7028e-04, 0.0000e+00},
	}
	rcdeinlist := [][]float64{
		{1.0000e+00},
		{1.0000e+00},
		{1.0000e+00, 1.0000e+00},
		{1.0000e+00, 1.0000e+00},
		{1.0000e+00, 1.0000e+00},
		{6.9855e-01, 9.7611e-08, 9.7611e-08, 1.0000e-07, 1.0000e-07, 4.0659e-01},
		{1.1244e-05, 1.1244e-05, 1.0807e-05, 1.0807e-05},
		{1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00},
		{1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00},
		{2.4074e-35, 2.4074e-35, 2.4074e-35, 2.4074e-35, 2.4074e-35, 2.4074e-35},
		{2.4074e-35, 2.4074e-35, 2.4074e-35, 2.4074e-35, 2.4074e-35, 2.4074e-35},
		{1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00},
		{8.7105e-01, 8.7105e-01, 9.8194e-01, 9.7723e-01},
		{2.7888e-01, 3.5427e-01, 3.4558e-01, 5.4661e-01, 5.4661e-01, 9.6635e-01},
		{2.4000e-11, 6.0000e-12, 4.0000e-12, 6.0000e-12, 2.4000e-11},
		{4.7781e-01, 4.7781e-01, 4.6034e-01, 4.7500e-01, 4.7500e-01, 2.9729e-01, 4.2947e-01, 4.2947e-01, 7.0777e-01, 9.7257e-01},
		{8.1913e-01, 8.1913e-01, 8.0499e-01, 9.8222e-01},
		{6.7909e-01, 7.2671e-01, 3.9757e-01, 3.9757e-01, 4.2334e-01, 4.2334e-01},
		{6.4005e-01, 6.4005e-01, 6.9083e-01, 5.3435e-01, 5.3435e-01, 2.9802e-01, 7.3193e-01, 7.3193e-01, 3.0743e-01, 8.5623e-01},
		{4.7136e-01, 4.7136e-01, 8.0960e-01, 8.0960e-01},
		{4.7091e-01, 3.6889e-01, 3.6889e-01, 5.8849e-01, 5.8849e-01, 6.0845e-01},
		{3.7485e-07, 9.8979e-09, 1.0098e-08, 1.4996e-06},
		{7.3030e-01, 7.2801e-01, 8.2763e-01},
		{5.7839e-01, 5.7839e-01, 4.9641e-01, 7.0039e-01, 7.0039e-01, 9.2518e-01},
		{7.5301e-01, 6.0479e-01, 5.4665e-01, 4.2771e-01, 6.6370e-01, 5.7388e-01},
		{8.6603e-01, 5.0000e-01, 2.9582e-31, 2.9582e-31, 5.0000e-01, 8.6603e-01},
		{1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00},
		{4.0841e-01, 3.7927e-01, 6.2793e-01, 8.1156e-01, 9.7087e-01, 8.4325e-01},
		{2.8690e-06, 1.5885e-06, 1.5885e-06, 6.5757e-07, 1.8324e-06, 6.8640e-05, 4.6255e-03, 1.4447e-01, 5.8447e-01, 3.1823e-01, 2.0079e-01, 3.0424e-01},
		{8.2295e-01, 7.2281e-01, 6.2854e-01, 6.2854e-01, 7.2281e-01, 8.2295e-01},
		{8.9968e-01, 8.9968e-01, 9.6734e-01, 9.6734e-01, 9.7605e-01, 9.7605e-01},
		{3.9345e-01, 3.9345e-01, 2.8868e-01, 4.5013e-01, 4.5013e-01, 9.6305e-01},
		{5.3287e-01, 7.2342e-01, 7.3708e-01, 4.4610e-01, 4.2881e-01, 4.8057e-01},
		{1.2253e-01, 4.9811e-02, 3.6914e-02, 3.3328e-02, 3.3328e-02, 3.6914e-02, 4.9811e-02, 1.2253e-01},
		{1.7357e-04, 1.7486e-04, 1.7486e-04, 1.7635e-04, 1.7635e-04, 1.7623e-04},
		{1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00},
		{2.1768e-01, 3.9966e-04, 3.9966e-04, 3.9976e-04, 3.9976e-04},
		{6.4875e-05, 6.4875e-05, 5.4076e-02, 5.4076e-02, 8.6149e-01, 1.2425e-01},
		{3.6037e-02, 7.4283e-05, 1.4346e-04, 1.4346e-04, 6.7873e-05, 6.7873e-05, 9.2779e-05, 2.7358e-04, 2.7358e-04, 9.2696e-05},
	}
	rcdvinlist := [][]float64{
		{0.0000e+00},
		{1.0000e+00},
		{0.0000e+00, 0.0000e+00},
		{4.0000e+00, 4.0000e+00},
		{4.0000e+00, 4.0000e+00},
		{2.2823e+00, 5.0060e-14, 5.0060e-14, 9.4094e-14, 9.4094e-14, 1.5283e+00},
		{3.1179e-05, 3.1179e-05, 2.9981e-05, 2.9981e-05},
		{1.9722e-31, 1.9722e-31, 1.9722e-31, 1.9722e-31, 1.9722e-31},
		{1.9722e-31, 1.9722e-31, 1.9722e-31, 1.9722e-31, 1.9722e-31},
		{2.4074e-35, 2.4074e-35, 2.4074e-35, 2.4074e-35, 2.4074e-35, 2.4074e-35},
		{2.4074e-35, 2.4074e-35, 2.4074e-35, 2.4074e-35, 2.4074e-35, 2.4074e-35},
		{1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00},
		{3.5073e-01, 3.5073e-01, 4.6989e-01, 1.5455e+00},
		{1.1793e-01, 6.8911e-02, 1.3489e-01, 1.7729e-01, 1.7729e-01, 1.8270e+00},
		{2.3952e-11, 5.9940e-12, 3.9920e-12, 5.9940e-12, 2.3952e-11},
		{1.5842e-01, 1.5842e-01, 1.9931e-01, 9.1686e-02, 9.1686e-02, 8.2469e-02, 3.9688e-02, 3.9688e-02, 1.5033e-01, 3.5548e+00},
		{7.7090e-01, 7.7090e-01, 4.9037e-01, 4.9037e-01},
		{6.7220e-01, 2.0436e-01, 4.9797e-01, 4.9797e-01, 1.9048e-01, 1.9048e-01},
		{4.1615e-01, 4.1615e-01, 2.5476e-01, 6.0878e-01, 6.0878e-01, 4.7530e-01, 6.2016e-01, 6.2016e-01, 4.1726e-01, 4.3175e-01},
		{7.1772e-01, 7.1772e-01, 5.1408e-01, 5.1408e-01},
		{8.5788e-01, 1.8909e-01, 1.8909e-01, 1.3299e+00, 1.3299e+00, 9.6725e-01},
		{3.6932e-07, 9.8493e-09, 1.0046e-08, 1.4773e-06},
		{4.0000e+00, 1.3726e-06, 2.2096e-06},
		{2.0310e-01, 2.0310e-01, 2.1574e-01, 4.1945e-02, 4.1945e-02, 1.7390e+00},
		{6.0533e-01, 2.8613e-01, 1.7376e-01, 3.0915e-01, 7.6443e-02, 8.9227e-02},
		{7.2597e-01, 2.6417e-01, 1.4600e-07, 6.2446e-08, 2.6417e-01, 3.7896e-01},
		{3.9290e-02, 3.9290e-02, 5.2120e-01, 5.2120e-01, 7.5089e-02, 7.5089e-02},
		{1.6605e-01, 3.0531e-02, 7.8195e-04, 7.2478e-05, 7.2478e-05, 1.8048e+00},
		{3.2094e-06, 9.9934e-07, 9.9934e-07, 7.8673e-07, 2.0796e-06, 6.1058e-05, 6.4028e-03, 1.9470e-01, 1.2016e+00, 1.4273e+00, 2.4358e+00, 5.6865e+00},
		{1.2318e+00, 7.5970e-01, 6.9666e-01, 6.9666e-01, 7.5970e-01, 1.2318e+00},
		{1.5236e+00, 1.5236e+00, 4.2793e-01, 4.2793e-01, 2.2005e-01, 2.2005e-01},
		{1.8415e-01, 1.8415e-01, 1.7003e-01, 2.9425e-01, 2.9425e-01, 2.2469e+00},
		{3.8557e-01, 9.1303e-02, 1.1032e-02, 1.2861e-02, 1.7378e-01, 3.9201e+02},
		{1.2978e-01, 8.0162e-02, 8.2942e-02, 1.3738e-01, 1.1171e-01, 7.2156e-02, 6.8352e-02, 1.1527e-01},
		{2.8157e-04, 1.6704e-04, 1.6704e-04, 1.6828e-04, 1.6828e-04, 3.0778e-04},
		{4.3333e+00, 4.3333e+00, 4.3333e+00, 9.1250e+00},
		{5.2263e-01, 6.0947e-03, 6.0947e-03, 6.0960e-03, 6.0960e-03},
		{5.0367e-04, 5.0367e-04, 2.3507e-01, 2.3507e-01, 5.4838e-07, 1.2770e-06},
		{7.9613e-02, 7.4025e-06, 6.7839e-07, 6.7839e-07, 5.7496e-06, 5.7496e-06, 2.6519e-06, 1.9407e-07, 1.9407e-07, 2.6477e-06},
	}

	//     Read in data from file to check accuracy of condition estimation.
	//     Assume input eigenvalues are sorted lexicographically (increasing
	//     by real part, then decreasing by imaginary part)
	jtype = 0
	for _i, n = range nlist {
		jtype = jtype + 1
		iseed[0] = jtype
		for i = 1; i <= n; i++ {
			for j = 1; j <= n; j++ {
				a.Set(i-1, j-1, alist[_i][(i-1)*(n)+j-1])
			}
		}
		for i = 1; i <= n; i++ {
			wr1.Set(i-1, wr1list[_i][i-1])
			wi1.Set(i-1, wi1list[_i][i-1])
			rcdein.Set(i-1, rcdeinlist[_i][i-1])
			rcdvin.Set(i-1, rcdvinlist[_i][i-1])
		}
		iinfo, err = dget23(true, 'N', 22, thresh, iseed, nounit, n, a, h, wr, wi, wr1, wi1, vl, vr, lre, rcondv, rcndv1, rcdvin, rconde, rcnde1, rcdein, scale, scale1, result, work, 6*n+2*pow(n, 2), &iwork)

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
			t.Fail()
			ntestf = ntestf + 1
		}
		if ntestf == 1 {
			fmt.Printf("\n %3s -- Real Eigenvalue-Eigenvector Decomposition Expert Driver\n Matrix types (see ddrvvx for details): \n", path)
			fmt.Printf("\n Special Matrices:\n  1=Zero matrix.                          5=Diagonal: geometr. spaced entries.\n  2=Identity matrix.                      6=Diagonal: clustered entries.\n  3=Transposed Jordan block.              7=Diagonal: large, evenly spaced.\n  4=Diagonal: evenly spaced entries.      8=Diagonal: small, evenly spaced.\n")
			fmt.Printf(" Dense, Non-Symmetric Matrices:\n  9=Well-cond., evenly spaced eigenvals. 14=Ill-cond., geomet. spaced eigenals.\n 10=Well-cond., geom. spaced eigenvals.  15=Ill-conditioned, clustered e.vals.\n 11=Well-conditioned, clustered e.vals.  16=Ill-cond., random complex \n 12=Well-cond., random complex           17=Ill-cond., large rand. complx \n 13=Ill-conditioned, evenly spaced.      18=Ill-cond., small rand. complx \n")
			fmt.Printf(" 19=Matrix with random O(1) entries.     21=Matrix with small random entries.\n 20=Matrix with large random entries.    22=Matrix read from input file\n\n")
			fmt.Printf(" Tests performed with test threshold =%8.2f\n\n 1 = | A VR - VR W | / ( n |A| ulp ) \n 2 = | transpose(A) VL - VL W | / ( n |A| ulp ) \n 3 = | |VR(i)| - 1 | / ulp \n 4 = | |VL(i)| - 1 | / ulp \n 5 = 0 if W same no matter if VR or VL computed, 1/ulp otherwise\n 6 = 0 if VR same no matter what else computed,  1/ulp otherwise\n 7 = 0 if VL same no matter what else computed,  1/ulp otherwise\n 8 = 0 if RCONDV same no matter what else computed,  1/ulp otherwise\n 9 = 0 if SCALE, ILO, IHI, ABNRM same no matter what else computed,  1/ulp otherwise\n 10 = | RCONDV - RCONDV(precomputed) | / cond(RCONDV),\n 11 = | RCONDE - RCONDE(precomputed) | / cond(RCONDE),\n", thresh)
			ntestf = 2
		}

		for j = 1; j <= 11; j++ {
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
