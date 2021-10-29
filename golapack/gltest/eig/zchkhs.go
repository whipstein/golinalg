package eig

import (
	"fmt"
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// zchkhs checks the nonsymmetric eigenvalue problem routines.
//
//            Zgehrd factors A as  U H U' , where ' means conjugate
//            transpose, H is hessenberg, and U is unitary.
//
//            ZUNGHR generates the unitary matrix U.
//
//            Zunmhr multiplies a matrix by the unitary matrix U.
//
//            Zhseqr factors H as  Z T Z' , where Z is unitary and T
//            is upper triangular.  It also computes the eigenvalues,
//            w(1), ..., w(n); we define a diagonal matrix W whose
//            (diagonal) entries are the eigenvalues.
//
//            Ztrevc computes the left eigenvector matrix L and the
//            right eigenvector matrix R for the matrix T.  The
//            columns of L are the complex conjugates of the left
//            eigenvectors of T.  The columns of R are the right
//            eigenvectors of T.  L is lower triangular, and R is
//            upper triangular.
//
//            Zhsein computes the left eigenvector matrix Y and the
//            right eigenvector matrix X for the matrix H.  The
//            columns of Y are the complex conjugates of the left
//            eigenvectors of H.  The columns of X are the right
//            eigenvectors of H.  Y is lower triangular, and X is
//            upper triangular.
//
//    When zchkhs is called, a number of matrix "sizes" ("n's") and a
//    number of matrix "types" are specified.  For each size ("n")
//    and each _type of matrix, one matrix will be generated and used
//    to test the nonsymmetric eigenroutines.  For each matrix, 14
//    tests will be performed:
//
//    (1)     | A - U H U**H | / ( |A| n ulp )
//
//    (2)     | I - UU**H | / ( n ulp )
//
//    (3)     | H - Z T Z**H | / ( |H| n ulp )
//
//    (4)     | I - ZZ**H | / ( n ulp )
//
//    (5)     | A - UZ H (UZ)**H | / ( |A| n ulp )
//
//    (6)     | I - UZ (UZ)**H | / ( n ulp )
//
//    (7)     | T(Z computed) - T(Z not computed) | / ( |T| ulp )
//
//    (8)     | W(Z computed) - W(Z not computed) | / ( |W| ulp )
//
//    (9)     | TR - RW | / ( |T| |R| ulp )
//
//    (10)    | L**H T - W**H L | / ( |T| |L| ulp )
//
//    (11)    | HX - XW | / ( |H| |X| ulp )
//
//    (12)    | Y**H H - W**H Y | / ( |H| |Y| ulp )
//
//    (13)    | AX - XW | / ( |A| |X| ulp )
//
//    (14)    | Y**H A - W**H Y | / ( |A| |Y| ulp )
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
//    (7)  Same as (4), but multiplied by math.Sqrt( overflow threshold )
//    (8)  Same as (4), but multiplied by math.Sqrt( underflow threshold )
//
//    (9)  A matrix of the form  U' T U, where U is unitary and
//         T has evenly spaced entries 1, ..., ULP with random complex
//         angles on the diagonal and random O(1) entries _ the upper
//         triangle.
//
//    (10) A matrix of the form  U' T U, where U is unitary and
//         T has geometrically spaced entries 1, ..., ULP with random
//         complex angles on the diagonal and random O(1) entries _
//         the upper triangle.
//
//    (11) A matrix of the form  U' T U, where U is unitary and
//         T has "clustered" entries 1, ULP,..., ULP with random
//         complex angles on the diagonal and random O(1) entries _
//         the upper triangle.
//
//    (12) A matrix of the form  U' T U, where U is unitary and
//         T has complex eigenvalues randomly chosen from
//         ULP < |z| < 1   and random O(1) entries _ the upper
//         triangle.
//
//    (13) A matrix of the form  X' T X, where X has condition
//         math.Sqrt( ULP ) and T has evenly spaced entries 1, ..., ULP
//         with random complex angles on the diagonal and random O(1)
//         entries _ the upper triangle.
//
//    (14) A matrix of the form  X' T X, where X has condition
//         math.Sqrt( ULP ) and T has geometrically spaced entries
//         1, ..., ULP with random complex angles on the diagonal
//         and random O(1) entries _ the upper triangle.
//
//    (15) A matrix of the form  X' T X, where X has condition
//         math.Sqrt( ULP ) and T has "clustered" entries 1, ULP,..., ULP
//         with random complex angles on the diagonal and random O(1)
//         entries _ the upper triangle.
//
//    (16) A matrix of the form  X' T X, where X has condition
//         math.Sqrt( ULP ) and T has complex eigenvalues randomly chosen
//         from   ULP < |z| < 1   and random O(1) entries _ the upper
//         triangle.
//
//    (17) Same as (16), but multiplied by math.Sqrt( overflow threshold )
//    (18) Same as (16), but multiplied by math.Sqrt( underflow threshold )
//
//    (19) Nonsymmetric matrix with random entries chosen from |z| < 1
//    (20) Same as (19), but multiplied by math.Sqrt( overflow threshold )
//    (21) Same as (19), but multiplied by math.Sqrt( underflow threshold )
func zchkhs(nsizes int, nn []int, ntypes int, dotype []bool, iseed []int, thresh float64, a, h, t1, t2, u, z, uz *mat.CMatrix, w1, w3 *mat.CVector, evectl, evectr, evecty, evectx, uu *mat.CMatrix, tau, work *mat.CVector, nwork int, rwork *mat.Vector, iwork []int, _select []bool, result *mat.Vector) (err error) {
	var badnn, match bool
	var cone, czero complex128
	var aninv, anorm, cond, conds, one, ovfl, rtovfl, rtulp, rtulpi, rtunfl, temp1, temp2, ulp, ulpinv, unfl, zero float64
	var i, ihi, iinfo, ilo, imode, itype, j, jcol, jj, jsize, jtype, k, maxtyp, mtypes, n, n1, nerrs, nmats, nmax, ntest, ntestt int
	cdumma := cvf(4)
	dumma := vf(4)
	idumma := make([]int, 1)
	ioldsd := make([]int, 4)
	kconds := []int{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0}
	kmagn := []int{1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 2, 3}
	kmode := []int{0, 0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 5, 4, 3, 1, 5, 5, 5, 4, 3, 1}
	ktype := []int{1, 2, 3, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 9}

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	maxtyp = 21

	//     Check for errors
	ntestt = 0

	badnn = false
	nmax = 0
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
	} else if a.Rows <= 1 || a.Rows < nmax {
		err = fmt.Errorf("a.Rows <= 1 || a.Rows < nmax: a.Rows=%v, nmax=%v", a.Rows, nmax)
	} else if u.Rows <= 1 || u.Rows < nmax {
		err = fmt.Errorf("u.Rows <= 1 || u.Rows < nmax: u.Rows=%v, nmax=%v", u.Rows, nmax)
	} else if 4*nmax*nmax+2 > nwork {
		err = fmt.Errorf("4*nmax*nmax+2 > nwork: nmax=%v, nwork=%v", nmax, nwork)
	}

	if err != nil {
		gltest.Xerbla2("zchkhs", err)
		return
	}

	//     Quick return if possible
	if nsizes == 0 || ntypes == 0 {
		return
	}

	//     More important constants
	unfl = golapack.Dlamch(SafeMinimum)
	ovfl = golapack.Dlamch(Overflow)
	unfl, ovfl = golapack.Dlabad(unfl, ovfl)
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)
	ulpinv = one / ulp
	rtunfl = math.Sqrt(unfl)
	rtovfl = math.Sqrt(ovfl)
	rtulp = math.Sqrt(ulp)
	rtulpi = one / rtulp

	//     Loop over sizes, types
	nerrs = 0
	nmats = 0

	for jsize = 1; jsize <= nsizes; jsize++ {
		n = nn[jsize-1]
		if n == 0 {
			goto label260
		}
		n1 = max(1, n)
		aninv = one / float64(n1)

		if nsizes != 1 {
			mtypes = min(maxtyp, ntypes)
		} else {
			mtypes = min(maxtyp+1, ntypes)
		}

		for jtype = 1; jtype <= mtypes; jtype++ {
			if !dotype[jtype-1] {
				goto label250
			}
			nmats = nmats + 1
			ntest = 0

			//           Save iseed _ case of an error.
			for j = 1; j <= 4; j++ {
				ioldsd[j-1] = iseed[j-1]
			}

			//           Initialize RESULT
			for j = 1; j <= 14; j++ {
				result.Set(j-1, zero)
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
			//       =5                 random log   hermitian, w/ eigenvalues
			//       =6                 random       general, w/ eigenvalues
			//       =7                              random diagonal
			//       =8                              random hermitian
			//       =9                              random general
			//       =10                             random triangular
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

			golapack.Zlaset(Full, a.Rows, n, czero, czero, a)
			iinfo = 0
			cond = ulpinv

			//           Special Matrices
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
						a.SetRe(jcol-1, jcol-1-1, one)
					}
				}

			} else if itype == 4 {
				//              Diagonal Matrix, [Eigen]values Specified
				err = matgen.Zlatmr(n, n, 'D', &iseed, 'N', work, imode, cond, cone, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, 0, 0, zero, anorm, 'N', a, &iwork)

			} else if itype == 5 {
				//              Hermitian, eigenvalues specified
				err = matgen.Zlatms(n, n, 'D', &iseed, 'H', rwork, imode, cond, anorm, n, n, 'N', a, work)

			} else if itype == 6 {
				//              General, eigenvalues specified
				if kconds[jtype-1] == 1 {
					conds = one
				} else if kconds[jtype-1] == 2 {
					conds = rtulpi
				} else {
					conds = zero
				}

				err = matgen.Zlatme(n, 'D', &iseed, work, imode, cond, cone, 'T', 'T', 'T', rwork, 4, conds, n, n, anorm, a, work.Off(n))

			} else if itype == 7 {
				//              Diagonal, random eigenvalues
				err = matgen.Zlatmr(n, n, 'D', &iseed, 'N', work, 6, one, cone, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, 0, 0, zero, anorm, 'N', a, &iwork)

			} else if itype == 8 {
				//              Hermitian, random eigenvalues
				err = matgen.Zlatmr(n, n, 'D', &iseed, 'H', work, 6, one, cone, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, n, n, zero, anorm, 'N', a, &iwork)

			} else if itype == 9 {
				//              General, random eigenvalues
				err = matgen.Zlatmr(n, n, 'D', &iseed, 'N', work, 6, one, cone, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, n, n, zero, anorm, 'N', a, &iwork)

			} else if itype == 10 {
				//              Triangular, random eigenvalues
				err = matgen.Zlatmr(n, n, 'D', &iseed, 'N', work, 6, one, cone, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, n, 0, zero, anorm, 'N', a, &iwork)

			} else {

				iinfo = 1
			}

			if iinfo != 0 || err != nil {
				fmt.Printf(" zchkhs: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Generator", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				return
			}

		label100:
			;

			//           Call Zgehrd to compute H and U, do tests.
			golapack.Zlacpy(Full, n, n, a, h)
			ntest = 1

			ilo = 1
			ihi = n

			if err = golapack.Zgehrd(n, ilo, ihi, h, work, work.Off(n), nwork-n); err != nil {
				result.Set(0, ulpinv)
				fmt.Printf(" zchkhs: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zgehrd", iinfo, n, jtype, ioldsd)
				goto label240
			}

			for j = 1; j <= n-1; j++ {
				uu.Set(j, j-1, czero)
				for i = j + 2; i <= n; i++ {
					u.Set(i-1, j-1, h.Get(i-1, j-1))
					uu.Set(i-1, j-1, h.Get(i-1, j-1))
					h.Set(i-1, j-1, czero)
				}
			}
			goblas.Zcopy(n-1, work.Off(0, 1), tau.Off(0, 1))
			if err = golapack.Zunghr(n, ilo, ihi, u, work, work.Off(n), nwork-n); err != nil {
				panic(err)
			}
			ntest = 2

			zhst01(n, ilo, ihi, a, h, u, work, nwork, rwork, result)

			//           Call Zhseqr to compute T1, T2 and Z, do tests.
			//
			//           Eigenvalues only (W3)
			golapack.Zlacpy(Full, n, n, h, t2)
			ntest = 3
			result.Set(2, ulpinv)

			if iinfo, err = golapack.Zhseqr('E', 'N', n, ilo, ihi, t2, w3, uz, work, nwork); err != nil || iinfo != 0 {
				fmt.Printf(" zchkhs: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zhseqr(E)", iinfo, n, jtype, ioldsd)
				if iinfo <= n+2 {
					goto label240
				}
			}

			//           Eigenvalues (W1) and Full Schur Form (T2)
			golapack.Zlacpy(Full, n, n, h, t2)

			if iinfo, err = golapack.Zhseqr('S', 'N', n, ilo, ihi, t2, w1, uz, work, nwork); err != nil || iinfo != 0 && iinfo <= n+2 {
				fmt.Printf(" zchkhs: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zhseqr(S)", iinfo, n, jtype, ioldsd)
				goto label240
			}

			//           Eigenvalues (W1), Schur Form (T1), and Schur Vectors (UZ)
			golapack.Zlacpy(Full, n, n, h, t1)
			golapack.Zlacpy(Full, n, n, u, uz)

			if iinfo, err = golapack.Zhseqr('S', 'V', n, ilo, ihi, t1, w1, uz, work, nwork); err != nil || iinfo != 0 && iinfo <= n+2 {
				fmt.Printf(" zchkhs: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zhseqr(V)", iinfo, n, jtype, ioldsd)
				goto label240
			}

			//           Compute Z = U' UZ
			if err = goblas.Zgemm(ConjTrans, NoTrans, n, n, n, cone, u, uz, czero, z); err != nil {
				panic(err)
			}
			ntest = 8

			//           Do Tests 3: | H - Z T Z' | / ( |H| n ulp )
			//                and 4: | I - Z Z' | / ( n ulp )
			zhst01(n, ilo, ihi, h, t1, z, work, nwork, rwork, result.Off(2))

			//           Do Tests 5: | A - UZ T (UZ)' | / ( |A| n ulp )
			//                and 6: | I - UZ (UZ)' | / ( n ulp )
			zhst01(n, ilo, ihi, a, t1, uz, work, nwork, rwork, result.Off(4))

			//           Do Test 7: | T2 - T1 | / ( |T| n ulp )
			result.Set(6, zget10(n, n, t2, t1, work, rwork))

			//           Do Test 8: | W3 - W1 | / ( max(|W1|,|W3|) ulp )
			temp1 = zero
			temp2 = zero
			for j = 1; j <= n; j++ {
				temp1 = math.Max(temp1, math.Max(w1.GetMag(j-1), w3.GetMag(j-1)))
				temp2 = math.Max(temp2, cmplx.Abs(w1.Get(j-1)-w3.Get(j-1)))
			}

			result.Set(7, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

			//           Compute the Left and Right Eigenvectors of T
			//
			//           Compute the Right eigenvector Matrix:
			ntest = 9
			result.Set(8, ulpinv)

			//           _select every other eigenvector
			for j = 1; j <= n; j++ {
				_select[j-1] = false
			}
			for j = 1; j <= n; j += 2 {
				_select[j-1] = true
			}
			if _, err = golapack.Ztrevc(Right, 'A', _select, n, t1, cdumma.CMatrix(u.Rows, opts), evectr, n, work, rwork); err != nil {
				fmt.Printf(" zchkhs: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Ztrevc(R,A)", iinfo, n, jtype, ioldsd)
				goto label240
			}

			//           Test 9:  | TR - RW | / ( |T| |R| ulp )
			zget22(NoTrans, NoTrans, NoTrans, n, t1, evectr, w1, work, rwork, dumma.Off(0))
			result.Set(8, dumma.Get(0))
			if dumma.Get(1) > thresh {
				fmt.Printf(" zchkhs: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         n=%6d, jtype=%6d, iseed=%5d\n", "Right", "Ztrevc", dumma.Get(1), n, jtype, ioldsd)
				err = fmt.Errorf(" zchkhs: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         n=%6d, jtype=%6d, iseed=%5d\n", "Right", "Ztrevc", dumma.Get(1), n, jtype, ioldsd)
			}

			//           Compute selected right eigenvectors and confirm that
			//           they agree with previous right eigenvectors
			if _, err = golapack.Ztrevc(Right, 'S', _select, n, t1, cdumma.CMatrix(u.Rows, opts), evectl, n, work, rwork); err != nil {
				fmt.Printf(" zchkhs: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Ztrevc(R,S)", iinfo, n, jtype, ioldsd)
				goto label240
			}

			k = 1
			match = true
			for j = 1; j <= n; j++ {
				if _select[j-1] {
					for jj = 1; jj <= n; jj++ {
						if evectr.Get(jj-1, j-1) != evectl.Get(jj-1, k-1) {
							match = false
							goto label180
						}
					}
					k = k + 1
				}
			}
		label180:
			;
			if !match {
				fmt.Printf(" zchkhs: Selected %s Eigenvectors from %s do not match other eigenvectors          n=%6d, jtype=%6d, iseed=%5d\n", "Right", "Ztrevc", n, jtype, ioldsd)
			}

			//           Compute the Left eigenvector Matrix:
			ntest = 10
			result.Set(9, ulpinv)
			if _, err = golapack.Ztrevc(Left, 'A', _select, n, t1, evectl, cdumma.CMatrix(u.Rows, opts), n, work, rwork); err != nil {
				fmt.Printf(" zchkhs: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Ztrevc(L,A)", iinfo, n, jtype, ioldsd)
				goto label240
			}

			//           Test 10:  | LT - WL | / ( |T| |L| ulp )
			zget22(ConjTrans, NoTrans, ConjTrans, n, t1, evectl, w1, work, rwork, dumma.Off(2))
			result.Set(9, dumma.Get(2))
			if dumma.Get(3) > thresh {
				fmt.Printf(" zchkhs: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         n=%6d, jtype=%6d, iseed=%5d\n", "Left", "Ztrevc", dumma.Get(3), n, jtype, ioldsd)
				err = fmt.Errorf(" zchkhs: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         n=%6d, jtype=%6d, iseed=%5d\n", "Left", "Ztrevc", dumma.Get(3), n, jtype, ioldsd)
			}

			//           Compute selected left eigenvectors and confirm that
			//           they agree with previous left eigenvectors
			if _, err = golapack.Ztrevc(Left, 'S', _select, n, t1, evectr, cdumma.CMatrix(u.Rows, opts), n, work, rwork); err != nil {
				fmt.Printf(" zchkhs: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Ztrevc(L,S)", iinfo, n, jtype, ioldsd)
				goto label240
			}

			k = 1
			match = true
			for j = 1; j <= n; j++ {
				if _select[j-1] {
					for jj = 1; jj <= n; jj++ {
						if evectl.Get(jj-1, j-1) != evectr.Get(jj-1, k-1) {
							match = false
							goto label210
						}
					}
					k = k + 1
				}
			}
		label210:
			;
			if !match {
				fmt.Printf(" zchkhs: Selected %s Eigenvectors from %s do not match other eigenvectors          n=%6d, jtype=%6d, iseed=%5d\n", "Left", "Ztrevc", n, jtype, ioldsd)
			}

			//           Call Zhsein for Right eigenvectors of H, do test 11
			ntest = 11
			result.Set(10, ulpinv)
			for j = 1; j <= n; j++ {
				_select[j-1] = true
			}

			if _, iinfo, err = golapack.Zhsein(Right, 'Q', 'N', _select, n, h, w3, cdumma.CMatrix(u.Rows, opts), evectx, n1, work, rwork, &iwork, &iwork); err != nil || iinfo != 0 {
				fmt.Printf(" zchkhs: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zhsein(R)", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					goto label240
				}
			} else {
				//              Test 11:  | HX - XW | / ( |H| |X| ulp )
				//
				//                        (from inverse iteration)
				zget22(NoTrans, NoTrans, NoTrans, n, h, evectx, w3, work, rwork, dumma.Off(0))
				if dumma.Get(0) < ulpinv {
					result.Set(10, dumma.Get(0)*aninv)
				}
				if dumma.Get(1) > thresh {
					fmt.Printf(" zchkhs: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         n=%6d, jtype=%6d, iseed=%5d\n", "Right", "Zhsein", dumma.Get(1), n, jtype, ioldsd)
					err = fmt.Errorf(" zchkhs: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         n=%6d, jtype=%6d, iseed=%5d\n", "Right", "Zhsein", dumma.Get(1), n, jtype, ioldsd)
				}
			}

			//           Call Zhsein for Left eigenvectors of H, do test 12
			ntest = 12
			result.Set(11, ulpinv)
			for j = 1; j <= n; j++ {
				_select[j-1] = true
			}

			if _, iinfo, err = golapack.Zhsein(Left, 'Q', 'N', _select, n, h, w3, evecty, cdumma.CMatrix(u.Rows, opts), n1, work, rwork, &iwork, &iwork); err != nil || iinfo != 0 {
				fmt.Printf(" zchkhs: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zhsein(L)", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					goto label240
				}
			} else {
				//              Test 12:  | YH - WY | / ( |H| |Y| ulp )
				//
				//                        (from inverse iteration)
				zget22(ConjTrans, NoTrans, ConjTrans, n, h, evecty, w3, work, rwork, dumma.Off(2))
				if dumma.Get(2) < ulpinv {
					result.Set(11, dumma.Get(2)*aninv)
				}
				if dumma.Get(3) > thresh {
					fmt.Printf(" zchkhs: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         n=%6d, jtype=%6d, iseed=%5d\n", "Left", "Zhsein", dumma.Get(3), n, jtype, ioldsd)
					err = fmt.Errorf(" zchkhs: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         n=%6d, jtype=%6d, iseed=%5d\n", "Left", "Zhsein", dumma.Get(3), n, jtype, ioldsd)
				}
			}

			//           Call Zunmhr for Right eigenvectors of A, do test 13
			ntest = 13
			result.Set(12, ulpinv)

			if err = golapack.Zunmhr(Left, NoTrans, n, n, ilo, ihi, uu, tau, evectx, work, nwork); err != nil {
				fmt.Printf(" zchkhs: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zunmhr(L)", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					goto label240
				}
			} else {
				//              Test 13:  | AX - XW | / ( |A| |X| ulp )
				//
				//                        (from inverse iteration)
				zget22(NoTrans, NoTrans, NoTrans, n, a, evectx, w3, work, rwork, dumma.Off(0))
				if dumma.Get(0) < ulpinv {
					result.Set(12, dumma.Get(0)*aninv)
				}
			}

			//           Call Zunmhr for Left eigenvectors of A, do test 14
			ntest = 14
			result.Set(13, ulpinv)

			if err = golapack.Zunmhr(Left, NoTrans, n, n, ilo, ihi, uu, tau, evecty, work, nwork); err != nil {
				fmt.Printf(" zchkhs: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zunmhr(L)", iinfo, n, jtype, ioldsd)
				if iinfo < 0 {
					goto label240
				}
			} else {
				//              Test 14:  | YA - WY | / ( |A| |Y| ulp )
				//
				//                        (from inverse iteration)
				zget22(ConjTrans, NoTrans, ConjTrans, n, a, evecty, w3, work, rwork, dumma.Off(2))
				if dumma.Get(2) < ulpinv {
					result.Set(13, dumma.Get(2)*aninv)
				}
			}

			//           End of Loop -- Check for RESULT(j) > THRESH
		label240:
			;

			ntestt = ntestt + ntest
			err = dlafts("Zhs", n, n, jtype, ntest, result, ioldsd, thresh, nerrs)

		label250:
		}
	label260:
	}

	//     Summary
	dlasum("Zhs", nerrs, ntestt)

	return
}
