package eig

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// dchkst2stg checks the symmetric eigenvalue problem routines
// using the 2-stage reduction techniques. Since the generation
// of Q or the vectors is not available in this release, we only
// compare the eigenvalue resulting when using the 2-stage to the
// one considered as reference using the standard 1-stage reduction
// Dsytrd. For that, we call the standard Dsytrd and compute D1 using
// Dsteqr, then we call the 2-stage DSYTRD_2STAGE with Upper and Lower
// and we compute D2 and D3 using Dsteqr and then we replaced tests
// 3 and 4 by tests 11 and 12. test 1 and 2 remain to verify that
// the 1-stage results are OK and can be trusted.
// This testing routine will converge to the DCHKST in the next
// release when vectors and generation of Q will be implemented.
//
//    Dsytrd factors A as  U S U' , where ' means transpose,
//    S is symmetric tridiagonal, and U is orthogonal.
//    Dsytrd can use either just the lower or just the upper triangle
//    of A; dchkst2stg checks both cases.
//    U is represented as a product of Householder
//    transformations, whose vectors are stored in the first
//    n-1 columns of V, and whose scale factors are in TAU.
//
//    Dsptrd does the same as Dsytrd, except that A and V are stored
//    in "packed" format.
//
//    Dorgtr constructs the matrix U from the contents of V and TAU.
//
//    Dopgtr constructs the matrix U from the contents of VP and TAU.
//
//    Dsteqr factors S as  Z D1 Z' , where Z is the orthogonal
//    matrix of eigenvectors and D1 is a diagonal matrix with
//    the eigenvalues on the diagonal.  D2 is the matrix of
//    eigenvalues computed when Z is not computed.
//
//    DSTERF computes D3, the matrix of eigenvalues, by the
//    PWK method, which does not yield eigenvectors.
//
//    Dpteqr factors S as  Z4 D4 Z4' , for a
//    symmetric positive definite tridiagonal matrix.
//    D5 is the matrix of eigenvalues computed when Z is not
//    computed.
//
//    Dstebz computes selected eigenvalues.  WA1, WA2, and
//    WA3 will denote eigenvalues computed to high
//    absolute accuracy, with different range options.
//    WR will denote eigenvalues computed to high relative
//    accuracy.
//
//    DSTEIN computes Y, the eigenvectors of S, given the
//    eigenvalues.
//
//    Dstedc factors S as Z D1 Z' , where Z is the orthogonal
//    matrix of eigenvectors and D1 is a diagonal matrix with
//    the eigenvalues on the diagonal ('I' option). It may also
//    update an input orthogonal matrix, usually the output
//    from Dsytrd/Dorgtr or Dsptrd/Dopgtr ('V' option). It may
//    also just compute eigenvalues ('N' option).
//
//    Dstemr factors S as Z D1 Z' , where Z is the orthogonal
//    matrix of eigenvectors and D1 is a diagonal matrix with
//    the eigenvalues on the diagonal ('I' option).  Dstemr
//    uses the Relatively Robust Representation whenever possible.
//
// When dchkst2stg is called, a number of matrix "sizes" ("n's") and a
// number of matrix "types" are specified.  For each size ("n")
// and each type of matrix, one matrix will be generated and used
// to test the symmetric eigenroutines.  For each matrix, a number
// of tests will be performed:
//
// (1)     | A - V S V' | / ( |A| n ulp ) Dsytrd( UPLO='U', ... )
//
// (2)     | I - UV' | / ( n ulp )        Dorgtr( UPLO='U', ... )
//
// (3)     | A - V S V' | / ( |A| n ulp ) Dsytrd( UPLO='L', ... )
//         replaced by | D1 - D2 | / ( |D1| ulp ) where D1 is the
//         eigenvalue matrix computed using S and D2 is the
//         eigenvalue matrix computed using S_2stage the output of
//         DSYTRD_2STAGE("N", "U",....). D1 and D2 are computed
//         via Dsteqr('N',...)
//
// (4)     | I - UV' | / ( n ulp )        Dorgtr( UPLO='L', ... )
//         replaced by | D1 - D3 | / ( |D1| ulp ) where D1 is the
//         eigenvalue matrix computed using S and D3 is the
//         eigenvalue matrix computed using S_2stage the output of
//         DSYTRD_2STAGE("N", "L",....). D1 and D3 are computed
//         via Dsteqr('N',...)
//
// (5-8)   Same as 1-4, but for Dsptrd and Dopgtr.
//
// (9)     | S - Z D Z' | / ( |S| n ulp ) Dsteqr('V',...)
//
// (10)    | I - ZZ' | / ( n ulp )        Dsteqr('V',...)
//
// (11)    | D1 - D2 | / ( |D1| ulp )        Dsteqr('N',...)
//
// (12)    | D1 - D3 | / ( |D1| ulp )        DSTERF
//
// (13)    0 if the true eigenvalues (computed by sturm count)
//         of S are within THRESH of
//         those in D1.  2*THRESH if they are not.  (Tested using
//         DSTECH)
//
// For S positive definite,
//
// (14)    | S - Z4 D4 Z4' | / ( |S| n ulp ) Dpteqr('V',...)
//
// (15)    | I - Z4 Z4' | / ( n ulp )        Dpteqr('V',...)
//
// (16)    | D4 - D5 | / ( 100 |D4| ulp )       Dpteqr('N',...)
//
// When S is also diagonally dominant by the factor gamma < 1,
//
// (17)    math.Max | D4(i) - WR(i) | / ( |D4(i)| omega ) ,
//          i
//         omega = 2 (2n-1) ULP (1 + 8 gamma**2) / (1 - gamma)**4
//                                              Dstebz( 'A', 'E', ...)
//
// (18)    | WA1 - D3 | / ( |D3| ulp )          Dstebz( 'A', 'E', ...)
//
// (19)    ( math.Max { min | WA2(i)-WA3(j) | } +
//            i     j
//           math.Max { min | WA3(i)-WA2(j) | } ) / ( |D3| ulp )
//            i     j
//                                              Dstebz( 'I', 'E', ...)
//
// (20)    | S - Y WA1 Y' | / ( |S| n ulp )  Dstebz, SSTEIN
//
// (21)    | I - Y Y' | / ( n ulp )          Dstebz, SSTEIN
//
// (22)    | S - Z D Z' | / ( |S| n ulp )    Dstedc('I')
//
// (23)    | I - ZZ' | / ( n ulp )           Dstedc('I')
//
// (24)    | S - Z D Z' | / ( |S| n ulp )    Dstedc('V')
//
// (25)    | I - ZZ' | / ( n ulp )           Dstedc('V')
//
// (26)    | D1 - D2 | / ( |D1| ulp )           Dstedc('V') and
//                                              Dstedc('N')
//
// Test 27 is disabled at the moment because Dstemr does not
// guarantee high relatvie accuracy.
//
// (27)    math.Max | D6(i) - WR(i) | / ( |D6(i)| omega ) ,
//          i
//         omega = 2 (2n-1) ULP (1 + 8 gamma**2) / (1 - gamma)**4
//                                              Dstemr('V', 'A')
//
// (28)    math.Max | D6(i) - WR(i) | / ( |D6(i)| omega ) ,
//          i
//         omega = 2 (2n-1) ULP (1 + 8 gamma**2) / (1 - gamma)**4
//                                              Dstemr('V', 'I')
//
// Tests 29 through 34 are disable at present because Dstemr
// does not handle partial spectrum requests.
//
// (29)    | S - Z D Z' | / ( |S| n ulp )    Dstemr('V', 'I')
//
// (30)    | I - ZZ' | / ( n ulp )           Dstemr('V', 'I')
//
// (31)    ( math.Max { min | WA2(i)-WA3(j) | } +
//            i     j
//           math.Max { min | WA3(i)-WA2(j) | } ) / ( |D3| ulp )
//            i     j
//         Dstemr('N', 'I') vs. SSTEMR('V', 'I')
//
// (32)    | S - Z D Z' | / ( |S| n ulp )    Dstemr('V', 'V')
//
// (33)    | I - ZZ' | / ( n ulp )           Dstemr('V', 'V')
//
// (34)    ( math.Max { min | WA2(i)-WA3(j) | } +
//            i     j
//           math.Max { min | WA3(i)-WA2(j) | } ) / ( |D3| ulp )
//            i     j
//         Dstemr('N', 'V') vs. SSTEMR('V', 'V')
//
// (35)    | S - Z D Z' | / ( |S| n ulp )    Dstemr('V', 'A')
//
// (36)    | I - ZZ' | / ( n ulp )           Dstemr('V', 'A')
//
// (37)    ( math.Max { min | WA2(i)-WA3(j) | } +
//            i     j
//           math.Max { min | WA3(i)-WA2(j) | } ) / ( |D3| ulp )
//            i     j
//         Dstemr('N', 'A') vs. SSTEMR('V', 'A')
//
// The "sizes" are specified by an array NN(1:NSIZES); the value of
// each element NN(j) specifies one size.
// The "types" are specified by a logical array DOTYPE( 1:NTYPES );
// if DOTYPE(j) is .TRUE., then matrix type "j" will be generated.
// Currently, the list of possible types is:
//
// (1)  The zero matrix.
// (2)  The identity matrix.
//
// (3)  A diagonal matrix with evenly spaced entries
//      1, ..., ULP  and random signs.
//      (ULP = (first number larger than 1) - 1 )
// (4)  A diagonal matrix with geometrically spaced entries
//      1, ..., ULP  and random signs.
// (5)  A diagonal matrix with "clustered" entries 1, ULP, ..., ULP
//      and random signs.
//
// (6)  Same as (4), but multiplied by SQRT( overflow threshold )
// (7)  Same as (4), but multiplied by SQRT( underflow threshold )
//
// (8)  A matrix of the form  U' D U, where U is orthogonal and
//      D has evenly spaced entries 1, ..., ULP with random signs
//      on the diagonal.
//
// (9)  A matrix of the form  U' D U, where U is orthogonal and
//      D has geometrically spaced entries 1, ..., ULP with random
//      signs on the diagonal.
//
// (10) A matrix of the form  U' D U, where U is orthogonal and
//      D has "clustered" entries 1, ULP,..., ULP with random
//      signs on the diagonal.
//
// (11) Same as (8), but multiplied by SQRT( overflow threshold )
// (12) Same as (8), but multiplied by SQRT( underflow threshold )
//
// (13) Symmetric matrix with random entries chosen from (-1,1).
// (14) Same as (13), but multiplied by SQRT( overflow threshold )
// (15) Same as (13), but multiplied by SQRT( underflow threshold )
// (16) Same as (8), but diagonal elements are all positive.
// (17) Same as (9), but diagonal elements are all positive.
// (18) Same as (10), but diagonal elements are all positive.
// (19) Same as (16), but multiplied by SQRT( overflow threshold )
// (20) Same as (16), but multiplied by SQRT( underflow threshold )
// (21) A diagonally dominant tridiagonal matrix with geometrically
//      spaced diagonal entries 1, ..., ULP.
func dchkst2stg(nsizes int, nn []int, ntypes int, dotype []bool, iseed []int, thresh float64, nounit int, a *mat.Matrix, ap, sd, se, d1, d2, d3, d4, d5, wa1, wa2, wa3, wr *mat.Vector, u, v *mat.Matrix, vp, tau *mat.Vector, z *mat.Matrix, work *mat.Vector, lwork int, iwork []int, liwork int, result *mat.Vector) (err error) {
	var badnn, srange, srel, tryrac bool
	var abstol, aninv, anorm, cond, eight, half, hun, one, ovfl, rtovfl, rtunfl, temp1, temp2, temp3, temp4, ten, two, ulp, ulpinv, unfl, vl, vu, zero float64
	var i, iinfo, il, imode, itemp, itype, iu, j, jc, jr, jsize, jtype, lgn, lh, liwedc, log2ui, lw, lwedc, m, m2, m3, maxtyp, mtypes, n, nap, nblock, nerrs, nmats, nmax, ntest, ntestt int
	idumma := make([]int, 1)
	ioldsd := make([]int, 4)
	iseed2 := make([]int, 4)
	kmagn := make([]int, 21)
	kmode := make([]int, 21)
	ktype := make([]int, 21)

	dumma := vf(1)

	zero = 0.0
	one = 1.0
	two = 2.0
	eight = 8.0
	ten = 10.0
	hun = 100.0
	half = one / two
	maxtyp = 21
	srange = false
	srel = false

	ktype[0], ktype[1], ktype[2], ktype[3], ktype[4], ktype[5], ktype[6], ktype[7], ktype[8], ktype[9], ktype[10], ktype[11], ktype[12], ktype[13], ktype[14], ktype[15], ktype[16], ktype[17], ktype[18], ktype[19], ktype[20] = 1, 2, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 8, 8, 8, 9, 9, 9, 9, 9, 10
	kmagn[0], kmagn[1], kmagn[2], kmagn[3], kmagn[4], kmagn[5], kmagn[6], kmagn[7], kmagn[8], kmagn[9], kmagn[10], kmagn[11], kmagn[12], kmagn[13], kmagn[14], kmagn[15], kmagn[16], kmagn[17], kmagn[18], kmagn[19], kmagn[20] = 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3, 1, 1, 1, 2, 3, 1
	kmode[0], kmode[1], kmode[2], kmode[3], kmode[4], kmode[5], kmode[6], kmode[7], kmode[8], kmode[9], kmode[10], kmode[11], kmode[12], kmode[13], kmode[14], kmode[15], kmode[16], kmode[17], kmode[18], kmode[19], kmode[20] = 0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0, 4, 3, 1, 4, 4, 3

	//     Keep ftnchek happy
	idumma[0] = 1

	//     Check for errors
	ntestt = 0

	//     Important constants
	badnn = false
	tryrac = true
	nmax = 1
	for j = 1; j <= nsizes; j++ {
		nmax = max(nmax, nn[j-1])
		if nn[j-1] < 0 {
			badnn = true
		}
	}

	nblock = ilaenv(1, "Dsytrd", []byte("L"), nmax, -1, -1, -1)
	nblock = min(nmax, max(1, nblock))

	//     Check for errors
	if nsizes < 0 {
		err = fmt.Errorf("nsizes < 0: nsizes=%v", nsizes)
	} else if badnn {
		err = fmt.Errorf("badnn: nn=%v", nn)
	} else if ntypes < 0 {
		err = fmt.Errorf("ntypes < 0: ntypes=%v", ntypes)
	} else if a.Rows < nmax {
		err = fmt.Errorf("a.Rows < nmax: a.Rows=%v, nmax=%v", a.Rows, nmax)
	} else if u.Rows < nmax {
		err = fmt.Errorf("u.Rows < nmax: u.Rows=%v, nmax=%v", u.Rows, nmax)
	} else if 2*pow(max(2, nmax), 2) > lwork {
		err = fmt.Errorf("2*pow(max(2, nmax), 2) > lwork: nmax=%v, lwork=%v", nmax, lwork)
	}

	if err != nil {
		gltest.Xerbla2("dchkst2stg", err)
		return
	}

	//     Quick return if possible
	if nsizes == 0 || ntypes == 0 {
		return
	}

	//     More Important constants
	unfl = golapack.Dlamch(SafeMinimum)
	ovfl = one / unfl
	unfl, ovfl = golapack.Dlabad(unfl, ovfl)
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)
	ulpinv = one / ulp
	log2ui = int(math.Log(ulpinv) / math.Log(two))
	rtunfl = math.Sqrt(unfl)
	rtovfl = math.Sqrt(ovfl)

	//     Loop over sizes, types
	for i = 1; i <= 4; i++ {
		iseed2[i-1] = iseed[i-1]
	}
	nerrs = 0
	nmats = 0

	for jsize = 1; jsize <= nsizes; jsize++ {
		n = nn[jsize-1]
		if n > 0 {
			lgn = int(math.Log(float64(n)) / math.Log(two))
			if int(math.Pow(2, float64(lgn))) < n {
				lgn = lgn + 1
			}
			if int(math.Pow(2, float64(lgn))) < n {
				lgn = lgn + 1
			}
			lwedc = 1 + 4*n + 2*n*lgn + 4*pow(n, 2)
			liwedc = 6 + 6*n + 5*n*lgn
		} else {
			lwedc = 8
			liwedc = 12
		}
		nap = (n * (n + 1)) / 2
		aninv = one / float64(max(1, n))

		if nsizes != 1 {
			mtypes = min(maxtyp, ntypes)
		} else {
			mtypes = min(maxtyp+1, ntypes)
		}

		for jtype = 1; jtype <= mtypes; jtype++ {
			if !dotype[jtype-1] {
				goto label300
			}
			nmats = nmats + 1
			ntest = 0

			for j = 1; j <= 4; j++ {
				ioldsd[j-1] = iseed[j-1]
			}

			//           Compute "A"
			//
			//           Control parameters:
			//
			//               KMAGN  KMODE        KTYPE
			//           =1  O(1)   clustered 1  zero
			//           =2  large  clustered 2  identity
			//           =3  small  exponential  (none)
			//           =4         arithmetic   diagonal, (w/ eigenvalues)
			//           =5         random log   symmetric, w/ eigenvalues
			//           =6         random       (none)
			//           =7                      random diagonal
			//           =8                      random symmetric
			//           =9                      positive definite
			//           =10                     diagonally dominant tridiagonal
			if mtypes > maxtyp {
				goto label100
			}

			itype = ktype[jtype-1]
			imode = kmode[jtype-1]

			//           Compute norm
			switch kmagn[jtype-1] {
			case 1:
				goto label40
			case 2:
				goto label50
			case 3:
				goto label60
			}

		label40:
			;
			anorm = one
			goto label70

		label50:
			;
			anorm = (rtovfl * ulp) * aninv
			goto label70

		label60:
			;
			anorm = rtunfl * float64(n) * ulpinv
			goto label70

		label70:
			;
			golapack.Dlaset(Full, a.Rows, n, zero, zero, a)
			iinfo = 0
			if jtype <= 15 {
				cond = ulpinv
			} else {
				cond = ulpinv * aninv / ten
			}

			//           Special Matrices -- Identity & Jordan block
			//
			//              Zero
			if itype == 1 {
				iinfo = 0

			} else if itype == 2 {
				//              Identity
				for jc = 1; jc <= n; jc++ {
					a.Set(jc-1, jc-1, anorm)
				}

			} else if itype == 4 {
				//              Diagonal Matrix, [Eigen]values Specified
				iinfo, err = matgen.Dlatms(n, n, 'S', &iseed, 'S', work, imode, cond, anorm, 0, 0, 'N', a, work.Off(n))

			} else if itype == 5 {
				//              Symmetric, eigenvalues specified
				iinfo, err = matgen.Dlatms(n, n, 'S', &iseed, 'S', work, imode, cond, anorm, n, n, 'N', a, work.Off(n))

			} else if itype == 7 {
				//              Diagonal, random eigenvalues
				iinfo, err = matgen.Dlatmr(n, n, 'S', &iseed, 'S', work, 6, one, one, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, 0, 0, zero, anorm, 'N', a, &iwork)

			} else if itype == 8 {
				//              Symmetric, random eigenvalues
				iinfo, err = matgen.Dlatmr(n, n, 'S', &iseed, 'S', work, 6, one, one, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, n, n, zero, anorm, 'N', a, &iwork)

			} else if itype == 9 {
				//              Positive definite, eigenvalues specified.
				iinfo, err = matgen.Dlatms(n, n, 'S', &iseed, 'P', work, imode, cond, anorm, n, n, 'N', a, work.Off(n))

			} else if itype == 10 {
				//              Positive definite tridiagonal, eigenvalues specified.
				iinfo, err = matgen.Dlatms(n, n, 'S', &iseed, 'P', work, imode, cond, anorm, 1, 1, 'N', a, work.Off(n))
				for i = 2; i <= n; i++ {
					temp1 = math.Abs(a.Get(i-1-1, i-1)) / math.Sqrt(math.Abs(a.Get(i-1-1, i-1-1)*a.Get(i-1, i-1)))
					if temp1 > half {
						a.Set(i-1-1, i-1, half*math.Sqrt(math.Abs(a.Get(i-1-1, i-1-1)*a.Get(i-1, i-1))))
						a.Set(i-1, i-1-1, a.Get(i-1-1, i-1))
					}
				}

			} else {

				iinfo = 1
			}

			if iinfo != 0 {
				fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Generator", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				return
			}

		label100:
			;

			//           Call Dsytrd and Dorgtr to compute S and U from
			//           upper triangle.
			golapack.Dlacpy(Upper, n, n, a, v)

			ntest = 1
			if err = golapack.Dsytrd(Upper, n, v, sd, se, tau, work, lwork); err != nil {
				fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsytrd(U)", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					return
				} else {
					result.Set(0, ulpinv)
					goto label280
				}
			}

			golapack.Dlacpy(Upper, n, n, v, u)

			ntest = 2
			if err = golapack.Dorgtr(Upper, n, u, tau, work, lwork); err != nil {
				fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dorgtr(U)", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				if iinfo < 0 {
					return
				} else {
					result.Set(1, ulpinv)
					goto label280
				}
			}

			//           Do tests 1 and 2
			dsyt21(2, Upper, n, 1, a, sd, se, u, v, tau, work, result.Off(0))
			dsyt21(3, Upper, n, 1, a, sd, se, u, v, tau, work, result.Off(1))

			//           Compute D1 the eigenvalues resulting from the tridiagonal
			//           form using the standard 1-stage algorithm and use it as a
			//           reference to compare with the 2-stage technique
			//
			//           Compute D1 from the 1-stage and used as reference for the
			//           2-stage
			goblas.Dcopy(n, sd.Off(0, 1), d1.Off(0, 1))
			if n > 0 {
				goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
			}

			if iinfo, err = golapack.Dsteqr('N', n, d1, work, work.MatrixOff(n, u.Rows, opts), work.Off(n)); err != nil || iinfo != 0 {
				fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsteqr(N)", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					return
				} else {
					result.Set(2, ulpinv)
					goto label280
				}
			}

			//           2-STAGE TRD Upper case is used to compute D2.
			//           Note to set SD and SE to zero to be sure not reusing
			//           the one from above. Compare it with D1 computed
			//           using the 1-stage.
			golapack.Dlaset(Full, n, 1, zero, zero, sd.Matrix(n, opts))
			golapack.Dlaset(Full, n, 1, zero, zero, se.Matrix(n, opts))
			golapack.Dlacpy(Upper, n, n, a, v)
			lh = max(1, 4*n)
			lw = lwork - lh
			if err = golapack.Dsytrd2stage('N', Upper, n, v, sd, se, tau, work, lh, work.Off(lh), lw); err != nil {
				panic(err)
			}

			//           Compute D2 from the 2-stage Upper case
			goblas.Dcopy(n, sd.Off(0, 1), d2.Off(0, 1))
			if n > 0 {
				goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
			}

			if iinfo, err = golapack.Dsteqr('N', n, d2, work, work.MatrixOff(n, u.Rows, opts), work.Off(n)); err != nil || iinfo != 0 {
				fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsteqr(N)", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					return
				} else {
					result.Set(2, ulpinv)
					goto label280
				}
			}

			//           2-STAGE TRD Lower case is used to compute D3.
			//           Note to set SD and SE to zero to be sure not reusing
			//           the one from above. Compare it with D1 computed
			//           using the 1-stage.
			golapack.Dlaset(Full, n, 1, zero, zero, sd.Matrix(n, opts))
			golapack.Dlaset(Full, n, 1, zero, zero, se.Matrix(n, opts))
			golapack.Dlacpy(Lower, n, n, a, v)
			if err = golapack.Dsytrd2stage('N', Lower, n, v, sd, se, tau, work, lh, work.Off(lh), lw); err != nil {
				panic(err)
			}

			//           Compute D3 from the 2-stage Upper case
			goblas.Dcopy(n, sd.Off(0, 1), d3.Off(0, 1))
			if n > 0 {
				goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
			}

			if iinfo, err = golapack.Dsteqr('N', n, d3, work, work.MatrixOff(n, u.Rows, opts), work.Off(n)); err != nil || iinfo != 0 {
				fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsteqr(N)", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					return
				} else {
					result.Set(3, ulpinv)
					goto label280
				}
			}

			//           Do Tests 3 and 4 which are similar to 11 and 12 but with the
			//           D1 computed using the standard 1-stage reduction as reference
			ntest = 4
			temp1 = zero
			temp2 = zero
			temp3 = zero
			temp4 = zero

			for j = 1; j <= n; j++ {
				temp1 = math.Max(temp1, math.Max(math.Abs(d1.Get(j-1)), math.Abs(d2.Get(j-1))))
				temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d2.Get(j-1)))
				temp3 = math.Max(temp3, math.Max(math.Abs(d1.Get(j-1)), math.Abs(d3.Get(j-1))))
				temp4 = math.Max(temp4, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
			}

			result.Set(2, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))
			result.Set(3, temp4/math.Max(unfl, ulp*math.Max(temp3, temp4)))

			//           Store the upper triangle of A in AP
			i = 0
			for jc = 1; jc <= n; jc++ {
				for jr = 1; jr <= jc; jr++ {
					i = i + 1
					ap.Set(i-1, a.Get(jr-1, jc-1))
				}
			}

			//           Call Dsptrd and Dopgtr to compute S and U from AP
			goblas.Dcopy(nap, ap.Off(0, 1), vp.Off(0, 1))

			ntest = 5
			if err = golapack.Dsptrd(Upper, n, vp, sd, se, tau); err != nil {
				fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsptrd(U)", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					return
				} else {
					result.Set(4, ulpinv)
					goto label280
				}
			}

			ntest = 6
			if err = golapack.Dopgtr(Upper, n, vp, tau, u, work); err != nil {
				fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dopgtr(U)", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				if iinfo < 0 {
					return
				} else {
					result.Set(5, ulpinv)
					goto label280
				}
			}

			//           Do tests 5 and 6
			dspt21(2, Upper, n, 1, ap, sd, se, u, vp, tau, work, result.Off(4))
			dspt21(3, Upper, n, 1, ap, sd, se, u, vp, tau, work, result.Off(5))

			//           Store the lower triangle of A in AP
			i = 0
			for jc = 1; jc <= n; jc++ {
				for jr = jc; jr <= n; jr++ {
					i = i + 1
					ap.Set(i-1, a.Get(jr-1, jc-1))
				}
			}

			//           Call Dsptrd and Dopgtr to compute S and U from AP
			goblas.Dcopy(nap, ap.Off(0, 1), vp.Off(0, 1))

			ntest = 7
			if err = golapack.Dsptrd(Lower, n, vp, sd, se, tau); err != nil {
				fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsptrd(L)", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					return
				} else {
					result.Set(6, ulpinv)
					goto label280
				}
			}

			ntest = 8
			if err = golapack.Dopgtr(Lower, n, vp, tau, u, work); err != nil {
				fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dopgtr(L)", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				if iinfo < 0 {
					return
				} else {
					result.Set(7, ulpinv)
					goto label280
				}
			}

			dspt21(2, Lower, n, 1, ap, sd, se, u, vp, tau, work, result.Off(6))
			dspt21(3, Lower, n, 1, ap, sd, se, u, vp, tau, work, result.Off(7))

			//           Call Dsteqr to compute D1, D2, and Z, do tests.
			//
			//           Compute D1 and Z
			goblas.Dcopy(n, sd.Off(0, 1), d1.Off(0, 1))
			if n > 0 {
				goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
			}
			golapack.Dlaset(Full, n, n, zero, one, z)

			ntest = 9
			if iinfo, err = golapack.Dsteqr('V', n, d1, work, z, work.Off(n)); err != nil || iinfo != 0 {
				fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsteqr(V)", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					return
				} else {
					result.Set(8, ulpinv)
					goto label280
				}
			}

			//           Compute D2
			goblas.Dcopy(n, sd.Off(0, 1), d2.Off(0, 1))
			if n > 0 {
				goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
			}

			ntest = 11
			if iinfo, err = golapack.Dsteqr('N', n, d2, work, work.MatrixOff(n, u.Rows, opts), work.Off(n)); err != nil || iinfo != 0 {
				fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsteqr(N)", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					return
				} else {
					result.Set(10, ulpinv)
					goto label280
				}
			}

			//           Compute D3 (using PWK method)
			goblas.Dcopy(n, sd.Off(0, 1), d3.Off(0, 1))
			if n > 0 {
				goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
			}

			ntest = 12
			if iinfo, err = golapack.Dsterf(n, d3, work); err != nil || iinfo != 0 {
				fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "DSTERF", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					return
				} else {
					result.Set(11, ulpinv)
					goto label280
				}
			}

			//           Do Tests 9 and 10
			dstt21(n, 0, sd, se, d1, dumma, z, work, result.Off(8))

			//           Do Tests 11 and 12
			temp1 = zero
			temp2 = zero
			temp3 = zero
			temp4 = zero

			for j = 1; j <= n; j++ {
				temp1 = math.Max(temp1, math.Max(math.Abs(d1.Get(j-1)), math.Abs(d2.Get(j-1))))
				temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d2.Get(j-1)))
				temp3 = math.Max(temp3, math.Max(math.Abs(d1.Get(j-1)), math.Abs(d3.Get(j-1))))
				temp4 = math.Max(temp4, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
			}

			result.Set(10, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))
			result.Set(11, temp4/math.Max(unfl, ulp*math.Max(temp3, temp4)))

			//           Do Test 13 -- Sturm Sequence Test of Eigenvalues
			//                         Go up by factors of two until it succeeds
			ntest = 13
			temp1 = thresh * (half - ulp)

			for j = 0; j <= log2ui; j++ {
				iinfo = dstech(n, sd, se, d1, temp1, work)
				if iinfo == 0 {
					goto label170
				}
				temp1 = temp1 * two
			}

		label170:
			;
			result.Set(12, temp1)

			//           For positive definite matrices ( jtype.GT.15 ) call Dpteqr
			//           and do tests 14, 15, and 16 .
			if jtype > 15 {
				//              Compute D4 and Z4
				goblas.Dcopy(n, sd.Off(0, 1), d4.Off(0, 1))
				if n > 0 {
					goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
				}
				golapack.Dlaset(Full, n, n, zero, one, z)

				ntest = 14
				if iinfo, err = golapack.Dpteqr('V', n, d4, work, z, work.Off(n)); err != nil || iinfo != 0 {
					fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dpteqr(V)", iinfo, n, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(13, ulpinv)
						goto label280
					}
				}

				//              Do Tests 14 and 15
				dstt21(n, 0, sd, se, d4, dumma, z, work, result.Off(13))

				//              Compute D5
				goblas.Dcopy(n, sd.Off(0, 1), d5.Off(0, 1))
				if n > 0 {
					goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
				}

				ntest = 16
				if iinfo, err = golapack.Dpteqr('N', n, d5, work, z, work.Off(n)); err != nil || iinfo != 0 {
					fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dpteqr(N)", iinfo, n, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(15, ulpinv)
						goto label280
					}
				}

				//              Do Test 16
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = math.Max(temp1, math.Max(math.Abs(d4.Get(j-1)), math.Abs(d5.Get(j-1))))
					temp2 = math.Max(temp2, math.Abs(d4.Get(j-1)-d5.Get(j-1)))
				}

				result.Set(15, temp2/math.Max(unfl, hun*ulp*math.Max(temp1, temp2)))
			} else {
				result.Set(13, zero)
				result.Set(14, zero)
				result.Set(15, zero)
			}

			//           Call Dstebz with different options and do tests 17-18.
			//
			//              If S is positive definite and diagonally dominant,
			//              ask for all eigenvalues with high relative accuracy.
			vl = zero
			vu = zero
			il = 0
			iu = 0
			if jtype == 21 {
				ntest = 17
				abstol = unfl + unfl
				if m, _, iinfo, err = golapack.Dstebz('A', 'E', n, vl, vu, il, iu, abstol, sd, se, wr, &iwork, toSlice(&iwork, n), work, toSlice(&iwork, 2*n)); err != nil || iinfo != 0 {
					fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstebz(A,rel)", iinfo, n, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(16, ulpinv)
						goto label280
					}
				}

				//              Do test 17
				temp2 = two * (two*float64(n) - one) * ulp * (one + eight*math.Pow(half, 2)) / math.Pow(one-half, 4)

				temp1 = zero
				for j = 1; j <= n; j++ {
					temp1 = math.Max(temp1, math.Abs(d4.Get(j-1)-wr.Get(n-j))/(abstol+math.Abs(d4.Get(j-1))))
				}

				result.Set(16, temp1/temp2)
			} else {
				result.Set(16, zero)
			}

			//           Now ask for all eigenvalues with high absolute accuracy.
			ntest = 18
			abstol = unfl + unfl
			if m, _, iinfo, err = golapack.Dstebz('A', 'E', n, vl, vu, il, iu, abstol, sd, se, wa1, &iwork, toSlice(&iwork, n), work, toSlice(&iwork, 2*n)); err != nil || iinfo != 0 {
				fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstebz(A)", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					return
				} else {
					result.Set(17, ulpinv)
					goto label280
				}
			}

			//           Do test 18
			temp1 = zero
			temp2 = zero
			for j = 1; j <= n; j++ {
				temp1 = math.Max(temp1, math.Max(math.Abs(d3.Get(j-1)), math.Abs(wa1.Get(j-1))))
				temp2 = math.Max(temp2, math.Abs(d3.Get(j-1)-wa1.Get(j-1)))
			}

			result.Set(17, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

			//           Choose random values for IL and IU, and ask for the
			//           IL-th through IU-th eigenvalues.
			ntest = 19
			if n <= 1 {
				il = 1
				iu = n
			} else {
				il = 1 + (n-1)*int(matgen.Dlarnd(1, &iseed2))
				iu = 1 + (n-1)*int(matgen.Dlarnd(1, &iseed2))
				if iu < il {
					itemp = iu
					iu = il
					il = itemp
				}
			}

			if m2, _, iinfo, err = golapack.Dstebz('I', 'E', n, vl, vu, il, iu, abstol, sd, se, wa2, &iwork, toSlice(&iwork, n), work, toSlice(&iwork, 2*n)); err != nil || iinfo != 0 {
				fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstebz(I)", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					return
				} else {
					result.Set(18, ulpinv)
					goto label280
				}
			}

			//           Determine the values VL and VU of the IL-th and IU-th
			//           eigenvalues and ask for all eigenvalues in this range.
			if n > 0 {
				if il != 1 {
					vl = wa1.Get(il-1) - math.Max(half*(wa1.Get(il-1)-wa1.Get(il-1-1)), math.Max(ulp*anorm, two*rtunfl))
				} else {
					vl = wa1.Get(0) - math.Max(half*(wa1.Get(n-1)-wa1.Get(0)), math.Max(ulp*anorm, two*rtunfl))
				}
				if iu != n {
					vu = wa1.Get(iu-1) + math.Max(half*(wa1.Get(iu)-wa1.Get(iu-1)), math.Max(ulp*anorm, two*rtunfl))
				} else {
					vu = wa1.Get(n-1) + math.Max(half*(wa1.Get(n-1)-wa1.Get(0)), math.Max(ulp*anorm, two*rtunfl))
				}
			} else {
				vl = zero
				vu = one
			}

			if m3, _, iinfo, err = golapack.Dstebz('V', 'E', n, vl, vu, il, iu, abstol, sd, se, wa3, &iwork, toSlice(&iwork, n), work, toSlice(&iwork, 2*n)); err != nil || iinfo != 0 {
				fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstebz(V)", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					return
				} else {
					result.Set(18, ulpinv)
					goto label280
				}
			}

			if m3 == 0 && n != 0 {
				result.Set(18, ulpinv)
				goto label280
			}

			//           Do test 19
			temp1 = dsxt1(1, wa2, m2, wa3, m3, abstol, ulp, unfl)
			temp2 = dsxt1(1, wa3, m3, wa2, m2, abstol, ulp, unfl)
			if n > 0 {
				temp3 = math.Max(math.Abs(wa1.Get(n-1)), math.Abs(wa1.Get(0)))
			} else {
				temp3 = zero
			}

			result.Set(18, (temp1+temp2)/math.Max(unfl, temp3*ulp))

			//           Call DSTEIN to compute eigenvectors corresponding to
			//           eigenvalues in WA1.  (First call Dstebz again, to make sure
			//           it returns these eigenvalues in the correct order.)
			ntest = 21
			if m, _, iinfo, err = golapack.Dstebz('A', 'B', n, vl, vu, il, iu, abstol, sd, se, wa1, &iwork, toSlice(&iwork, n), work, toSlice(&iwork, 2*n)); err != nil || iinfo != 0 {
				fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstebz(A,B)", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					return
				} else {
					result.Set(19, ulpinv)
					result.Set(20, ulpinv)
					goto label280
				}
			}

			if iinfo, err = golapack.Dstein(n, sd, se, m, wa1, &iwork, toSlice(&iwork, n), z, work, toSlice(&iwork, 2*n), toSlice(&iwork, 3*n)); err != nil || iinfo != 0 {
				fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "DSTEIN", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				if iinfo < 0 {
					return
				} else {
					result.Set(19, ulpinv)
					result.Set(20, ulpinv)
					goto label280
				}
			}

			//           Do tests 20 and 21
			dstt21(n, 0, sd, se, wa1, dumma, z, work, result.Off(19))

			//           Call Dstedc(I) to compute D1 and Z, do tests.
			//
			//           Compute D1 and Z
			goblas.Dcopy(n, sd.Off(0, 1), d1.Off(0, 1))
			if n > 0 {
				goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
			}
			golapack.Dlaset(Full, n, n, zero, one, z)

			ntest = 22
			if iinfo, err = golapack.Dstedc('I', n, d1, work, z, work.Off(n), lwedc-n, &iwork, liwedc); err != nil || iinfo != 0 {
				fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstedc(I)", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					return
				} else {
					result.Set(21, ulpinv)
					goto label280
				}
			}

			//           Do Tests 22 and 23
			dstt21(n, 0, sd, se, d1, dumma, z, work, result.Off(21))

			//           Call Dstedc(V) to compute D1 and Z, do tests.
			//
			//           Compute D1 and Z
			goblas.Dcopy(n, sd.Off(0, 1), d1.Off(0, 1))
			if n > 0 {
				goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
			}
			golapack.Dlaset(Full, n, n, zero, one, z)

			ntest = 24
			if iinfo, err = golapack.Dstedc('V', n, d1, work, z, work.Off(n), lwedc-n, &iwork, liwedc); err != nil || iinfo != 0 {
				fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstedc(V)", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					return
				} else {
					result.Set(23, ulpinv)
					goto label280
				}
			}

			//           Do Tests 24 and 25
			dstt21(n, 0, sd, se, d1, dumma, z, work, result.Off(23))

			//           Call Dstedc(N) to compute D2, do tests.
			//
			//           Compute D2
			goblas.Dcopy(n, sd.Off(0, 1), d2.Off(0, 1))
			if n > 0 {
				goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
			}
			golapack.Dlaset(Full, n, n, zero, one, z)

			ntest = 26
			if iinfo, err = golapack.Dstedc('N', n, d2, work, z, work.Off(n), lwedc-n, &iwork, liwedc); err != nil || iinfo != 0 {
				fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstedc(N)", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					return
				} else {
					result.Set(25, ulpinv)
					goto label280
				}
			}

			//           Do Test 26
			temp1 = zero
			temp2 = zero

			for j = 1; j <= n; j++ {
				temp1 = math.Max(temp1, math.Max(math.Abs(d1.Get(j-1)), math.Abs(d2.Get(j-1))))
				temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d2.Get(j-1)))
			}

			result.Set(25, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

			//           Only test Dstemr if IEEE compliant
			if ilaenv(10, "Dstemr", []byte("VA"), 1, 0, 0, 0) == 1 && ilaenv(11, "Dstemr", []byte("VA"), 1, 0, 0, 0) == 1 {
				//           Call Dstemr, do test 27 (relative eigenvalue accuracy)
				//
				//              If S is positive definite and diagonally dominant,
				//              ask for all eigenvalues with high relative accuracy.
				vl = zero
				vu = zero
				il = 0
				iu = 0
				if jtype == 21 && srel {
					ntest = 27
					abstol = unfl + unfl
					if m, tryrac, iinfo, err = golapack.Dstemr('V', 'A', n, sd, se, vl, vu, il, iu, wr, z, n, &iwork, tryrac, work, lwork, toSlice(&iwork, 2*n), lwork-2*n); err != nil || iinfo != 0 {
						fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstemr(V,A,rel)", iinfo, n, jtype, ioldsd)
						if iinfo < 0 {
							return
						} else {
							result.Set(26, ulpinv)
							goto label270
						}
					}

					//              Do test 27
					temp2 = two * (two*float64(n) - one) * ulp * (one + eight*math.Pow(half, 2)) / math.Pow(one-half, 4)

					temp1 = zero
					for j = 1; j <= n; j++ {
						temp1 = math.Max(temp1, math.Abs(d4.Get(j-1)-wr.Get(n-j))/(abstol+math.Abs(d4.Get(j-1))))
					}

					result.Set(26, temp1/temp2)

					il = 1 + (n-1)*int(matgen.Dlarnd(1, &iseed2))
					iu = 1 + (n-1)*int(matgen.Dlarnd(1, &iseed2))
					if iu < il {
						itemp = iu
						iu = il
						il = itemp
					}

					if srange {
						ntest = 28
						abstol = unfl + unfl
						if m, tryrac, iinfo, err = golapack.Dstemr('V', 'I', n, sd, se, vl, vu, il, iu, wr, z, n, &iwork, tryrac, work, lwork, toSlice(&iwork, 2*n), lwork-2*n); err != nil || iinfo != 0 {
							fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstemr(V,I,rel)", iinfo, n, jtype, ioldsd)
							if iinfo < 0 {
								return
							} else {
								result.Set(27, ulpinv)
								goto label270
							}
						}

						//                 Do test 28
						temp2 = two * (two*float64(n) - one) * ulp * (one + eight*math.Pow(half, 2)) / math.Pow(one-half, 4)

						temp1 = zero
						for j = il; j <= iu; j++ {
							temp1 = math.Max(temp1, math.Abs(wr.Get(j-il)-d4.Get(n-j))/(abstol+math.Abs(wr.Get(j-il))))
						}

						result.Set(27, temp1/temp2)
					} else {
						result.Set(27, zero)
					}
				} else {
					result.Set(26, zero)
					result.Set(27, zero)
				}

				//           Call Dstemr(V,I) to compute D1 and Z, do tests.
				//
				//           Compute D1 and Z
				goblas.Dcopy(n, sd.Off(0, 1), d5.Off(0, 1))
				if n > 0 {
					goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
				}
				golapack.Dlaset(Full, n, n, zero, one, z)

				if srange {
					ntest = 29
					il = 1 + (n-1)*int(matgen.Dlarnd(1, &iseed2))
					iu = 1 + (n-1)*int(matgen.Dlarnd(1, &iseed2))
					if iu < il {
						itemp = iu
						iu = il
						il = itemp
					}
					if m, tryrac, iinfo, err = golapack.Dstemr('V', 'I', n, d5, work, vl, vu, il, iu, d1, z, n, &iwork, tryrac, work.Off(n), lwork-n, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
						fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstemr(V,I)", iinfo, n, jtype, ioldsd)
						if iinfo < 0 {
							return
						} else {
							result.Set(28, ulpinv)
							goto label280
						}
					}

					//           Do Tests 29 and 30
					dstt22(n, m, 0, sd, se, d1, dumma, z, work.Matrix(m, opts), result.Off(28))

					//           Call Dstemr to compute D2, do tests.
					//
					//           Compute D2
					goblas.Dcopy(n, sd.Off(0, 1), d5.Off(0, 1))
					if n > 0 {
						goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
					}

					ntest = 31
					if m, tryrac, iinfo, err = golapack.Dstemr('N', 'I', n, d5, work, vl, vu, il, iu, d2, z, n, &iwork, tryrac, work.Off(n), lwork-n, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
						fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstemr(N,I)", iinfo, n, jtype, ioldsd)
						if iinfo < 0 {
							return
						} else {
							result.Set(30, ulpinv)
							goto label280
						}
					}

					//           Do Test 31
					temp1 = zero
					temp2 = zero

					for j = 1; j <= iu-il+1; j++ {
						temp1 = math.Max(temp1, math.Max(math.Abs(d1.Get(j-1)), math.Abs(d2.Get(j-1))))
						temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d2.Get(j-1)))
					}

					result.Set(30, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

					//           Call Dstemr(V,V) to compute D1 and Z, do tests.
					//
					//           Compute D1 and Z
					goblas.Dcopy(n, sd.Off(0, 1), d5.Off(0, 1))
					if n > 0 {
						goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
					}
					golapack.Dlaset(Full, n, n, zero, one, z)

					ntest = 32

					if n > 0 {
						if il != 1 {
							vl = d2.Get(il-1) - math.Max(half*(d2.Get(il-1)-d2.Get(il-1-1)), math.Max(ulp*anorm, two*rtunfl))
						} else {
							vl = d2.Get(0) - math.Max(half*(d2.Get(n-1)-d2.Get(0)), math.Max(ulp*anorm, two*rtunfl))
						}
						if iu != n {
							vu = d2.Get(iu-1) + math.Max(half*(d2.Get(iu)-d2.Get(iu-1)), math.Max(ulp*anorm, two*rtunfl))
						} else {
							vu = d2.Get(n-1) + math.Max(half*(d2.Get(n-1)-d2.Get(0)), math.Max(ulp*anorm, two*rtunfl))
						}
					} else {
						vl = zero
						vu = one
					}

					if m, tryrac, iinfo, err = golapack.Dstemr('V', 'V', n, d5, work, vl, vu, il, iu, d1, z, n, &iwork, tryrac, work.Off(n), lwork-n, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
						fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstemr(V,V)", iinfo, n, jtype, ioldsd)
						if iinfo < 0 {
							return
						} else {
							result.Set(31, ulpinv)
							goto label280
						}
					}

					//           Do Tests 32 and 33
					dstt22(n, m, 0, sd, se, d1, dumma, z, work.Matrix(m, opts), result.Off(31))

					//           Call Dstemr to compute D2, do tests.
					//
					//           Compute D2
					goblas.Dcopy(n, sd.Off(0, 1), d5.Off(0, 1))
					if n > 0 {
						goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
					}

					ntest = 34
					if m, tryrac, iinfo, err = golapack.Dstemr('N', 'V', n, d5, work, vl, vu, il, iu, d2, z, n, &iwork, tryrac, work.Off(n), lwork-n, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
						fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstemr(N,V)", iinfo, n, jtype, ioldsd)
						if iinfo < 0 {
							return
						} else {
							result.Set(33, ulpinv)
							goto label280
						}
					}

					//           Do Test 34
					temp1 = zero
					temp2 = zero

					for j = 1; j <= iu-il+1; j++ {
						temp1 = math.Max(temp1, math.Max(math.Abs(d1.Get(j-1)), math.Abs(d2.Get(j-1))))
						temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d2.Get(j-1)))
					}

					result.Set(33, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))
				} else {
					result.Set(28, zero)
					result.Set(29, zero)
					result.Set(30, zero)
					result.Set(31, zero)
					result.Set(32, zero)
					result.Set(33, zero)
				}

				//           Call Dstemr(V,A) to compute D1 and Z, do tests.
				//
				//           Compute D1 and Z
				goblas.Dcopy(n, sd.Off(0, 1), d5.Off(0, 1))
				if n > 0 {
					goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
				}

				ntest = 35

				if m, tryrac, iinfo, err = golapack.Dstemr('V', 'A', n, d5, work, vl, vu, il, iu, d1, z, n, &iwork, tryrac, work.Off(n), lwork-n, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
					fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstemr(V,A)", iinfo, n, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(34, ulpinv)
						goto label280
					}
				}

				//           Do Tests 35 and 36
				dstt22(n, m, 0, sd, se, d1, dumma, z, work.Matrix(m, opts), result.Off(34))

				//           Call Dstemr to compute D2, do tests.
				//
				//           Compute D2
				goblas.Dcopy(n, sd.Off(0, 1), d5.Off(0, 1))
				if n > 0 {
					goblas.Dcopy(n-1, se.Off(0, 1), work.Off(0, 1))
				}

				ntest = 37
				if m, tryrac, iinfo, err = golapack.Dstemr('N', 'A', n, d5, work, vl, vu, il, iu, d2, z, n, &iwork, tryrac, work.Off(n), lwork-n, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
					fmt.Printf(" dchkst2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstemr(N,A)", iinfo, n, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(36, ulpinv)
						goto label280
					}
				}

				//           Do Test 34
				temp1 = zero
				temp2 = zero

				for j = 1; j <= n; j++ {
					temp1 = math.Max(temp1, math.Max(math.Abs(d1.Get(j-1)), math.Abs(d2.Get(j-1))))
					temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d2.Get(j-1)))
				}

				result.Set(36, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))
			}
		label270:
			;
		label280:
			;
			ntestt = ntestt + ntest

			//           End of Loop -- Check for RESULT(j) > THRESH
			//
			//
			//           Print out tests which fail.
			for jr = 1; jr <= ntest; jr++ {
				if result.Get(jr-1) >= thresh {
					//                 If this is the first test to fail,
					//                 print a header to the data file.
					if nerrs == 0 {
						fmt.Printf("\n %3s -- Real Symmetric eigenvalue problem\n", "DST")
						fmt.Printf(" Matrix types (see dchkst2stg for details): \n")
						fmt.Printf("\n Special Matrices:\n  1=Zero matrix.                          5=Diagonal: clustered entries.\n  2=Identity matrix.                      6=Diagonal: large, evenly spaced.\n  3=Diagonal: evenly spaced entries.      7=Diagonal: small, evenly spaced.\n  4=Diagonal: geometr. spaced entries.\n")
						fmt.Printf(" Dense %s Matrices:\n  8=Evenly spaced eigenvals.             12=Small, evenly spaced eigenvals.\n  9=Geometrically spaced eigenvals.      13=Matrix with random O(1) entries.\n 10=Clustered eigenvalues.               14=Matrix with large random entries.\n 11=Large, evenly spaced eigenvals.      15=Matrix with small random entries.\n", "Symmetric")
						fmt.Printf(" 16=Positive definite, evenly spaced eigenvalues\n 17=Positive definite, geometrically spaced eigenvlaues\n 18=Positive definite, clustered eigenvalues\n 19=Positive definite, small evenly spaced eigenvalues\n 20=Positive definite, large evenly spaced eigenvalues\n 21=Diagonally dominant tridiagonal, geometrically spaced eigenvalues\n")

						//                    Tests performed
						fmt.Printf("\nTest performed:  see dchkst2stg for details.\n\n")
					}
					nerrs = nerrs + 1
					fmt.Printf(" n=%5d, seed=%4d, type %2d, test(%2d)=%10.3f\n", n, ioldsd, jtype, jr, result.Get(jr-1))
				}
			}
		label300:
		}
	}

	//     Summary
	dlasum("Dst", nerrs, ntestt)

	return
}
