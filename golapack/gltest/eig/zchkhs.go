package eig

import (
	"fmt"
	"math"
	"math/cmplx"
	"testing"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Zchkhs checks the nonsymmetric eigenvalue problem routines.
//
//            ZGEHRD factors A as  U H U' , where ' means conjugate
//            transpose, H is hessenberg, and U is unitary.
//
//            ZUNGHR generates the unitary matrix U.
//
//            ZUNMHR multiplies a matrix by the unitary matrix U.
//
//            ZHSEQR factors H as  Z T Z' , where Z is unitary and T
//            is upper triangular.  It also computes the eigenvalues,
//            w(1), ..., w(n); we define a diagonal matrix W whose
//            (diagonal) entries are the eigenvalues.
//
//            ZTREVC computes the left eigenvector matrix L and the
//            right eigenvector matrix R for the matrix T.  The
//            columns of L are the complex conjugates of the left
//            eigenvectors of T.  The columns of R are the right
//            eigenvectors of T.  L is lower triangular, and R is
//            upper triangular.
//
//            ZHSEIN computes the left eigenvector matrix Y and the
//            right eigenvector matrix X for the matrix H.  The
//            columns of Y are the complex conjugates of the left
//            eigenvectors of H.  The columns of X are the right
//            eigenvectors of H.  Y is lower triangular, and X is
//            upper triangular.
//
//    When ZCHKHS is called, a number of matrix "sizes" ("n's") and a
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
//         math.Sqrt( ULP ) and T has evenly spaced entries 1, ..., ULP
//         with random complex angles on the diagonal and random O(1)
//         entries in the upper triangle.
//
//    (14) A matrix of the form  X' T X, where X has condition
//         math.Sqrt( ULP ) and T has geometrically spaced entries
//         1, ..., ULP with random complex angles on the diagonal
//         and random O(1) entries in the upper triangle.
//
//    (15) A matrix of the form  X' T X, where X has condition
//         math.Sqrt( ULP ) and T has "clustered" entries 1, ULP,..., ULP
//         with random complex angles on the diagonal and random O(1)
//         entries in the upper triangle.
//
//    (16) A matrix of the form  X' T X, where X has condition
//         math.Sqrt( ULP ) and T has complex eigenvalues randomly chosen
//         from   ULP < |z| < 1   and random O(1) entries in the upper
//         triangle.
//
//    (17) Same as (16), but multiplied by math.Sqrt( overflow threshold )
//    (18) Same as (16), but multiplied by math.Sqrt( underflow threshold )
//
//    (19) Nonsymmetric matrix with random entries chosen from |z| < 1
//    (20) Same as (19), but multiplied by math.Sqrt( overflow threshold )
//    (21) Same as (19), but multiplied by math.Sqrt( underflow threshold )
func Zchkhs(nsizes *int, nn *[]int, ntypes *int, dotype *[]bool, iseed *[]int, thresh *float64, nounit *int, a *mat.CMatrix, lda *int, h, t1, t2, u *mat.CMatrix, ldu *int, z, uz *mat.CMatrix, w1, w3 *mat.CVector, evectl, evectr, evecty, evectx, uu *mat.CMatrix, tau, work *mat.CVector, nwork *int, rwork *mat.Vector, iwork *[]int, _select *[]bool, result *mat.Vector, info *int, t *testing.T) {
	var badnn, match bool
	var cone, czero complex128
	var aninv, anorm, cond, conds, one, ovfl, rtovfl, rtulp, rtulpi, rtunfl, temp1, temp2, ulp, ulpinv, unfl, zero float64
	var i, ihi, iinfo, ilo, imode, in, itype, j, jcol, jj, jsize, jtype, k, maxtyp, mtypes, n, n1, nerrs, nmats, nmax, ntest, ntestt int
	cdumma := cvf(4)
	dumma := vf(4)
	idumma := make([]int, 1)
	ioldsd := make([]int, 4)
	kconds := make([]int, 21)
	kmagn := make([]int, 21)
	kmode := make([]int, 21)
	ktype := make([]int, 21)

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	maxtyp = 21

	ktype[0], ktype[1], ktype[2], ktype[3], ktype[4], ktype[5], ktype[6], ktype[7], ktype[8], ktype[9], ktype[10], ktype[11], ktype[12], ktype[13], ktype[14], ktype[15], ktype[16], ktype[17], ktype[18], ktype[19], ktype[20] = 1, 2, 3, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 9
	kmagn[0], kmagn[1], kmagn[2], kmagn[3], kmagn[4], kmagn[5], kmagn[6], kmagn[7], kmagn[8], kmagn[9], kmagn[10], kmagn[11], kmagn[12], kmagn[13], kmagn[14], kmagn[15], kmagn[16], kmagn[17], kmagn[18], kmagn[19], kmagn[20] = 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 2, 3
	kmode[0], kmode[1], kmode[2], kmode[3], kmode[4], kmode[5], kmode[6], kmode[7], kmode[8], kmode[9], kmode[10], kmode[11], kmode[12], kmode[13], kmode[14], kmode[15], kmode[16], kmode[17], kmode[18], kmode[19], kmode[20] = 0, 0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 5, 4, 3, 1, 5, 5, 5, 4, 3, 1
	kconds[0], kconds[1], kconds[2], kconds[3], kconds[4], kconds[5], kconds[6], kconds[7], kconds[8], kconds[9], kconds[10], kconds[11], kconds[12], kconds[13], kconds[14], kconds[15], kconds[16], kconds[17], kconds[18], kconds[19], kconds[20] = 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0

	//     Check for errors
	ntestt = 0
	(*info) = 0

	badnn = false
	nmax = 0
	for j = 1; j <= (*nsizes); j++ {
		nmax = maxint(nmax, (*nn)[j-1])
		if (*nn)[j-1] < 0 {
			badnn = true
		}
	}

	//     Check for errors
	if (*nsizes) < 0 {
		(*info) = -1
	} else if badnn {
		(*info) = -2
	} else if (*ntypes) < 0 {
		(*info) = -3
	} else if (*thresh) < zero {
		(*info) = -6
	} else if (*lda) <= 1 || (*lda) < nmax {
		(*info) = -9
	} else if (*ldu) <= 1 || (*ldu) < nmax {
		(*info) = -14
	} else if 4*nmax*nmax+2 > (*nwork) {
		(*info) = -26
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZCHKHS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*nsizes) == 0 || (*ntypes) == 0 {
		return
	}

	//     More important constants
	unfl = golapack.Dlamch(SafeMinimum)
	ovfl = golapack.Dlamch(Overflow)
	golapack.Dlabad(&unfl, &ovfl)
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)
	ulpinv = one / ulp
	rtunfl = math.Sqrt(unfl)
	rtovfl = math.Sqrt(ovfl)
	rtulp = math.Sqrt(ulp)
	rtulpi = one / rtulp

	//     Loop over sizes, types
	nerrs = 0
	nmats = 0

	for jsize = 1; jsize <= (*nsizes); jsize++ {
		n = (*nn)[jsize-1]
		if n == 0 {
			goto label260
		}
		n1 = maxint(1, n)
		aninv = one / float64(n1)

		if (*nsizes) != 1 {
			mtypes = minint(maxtyp, *ntypes)
		} else {
			mtypes = minint(maxtyp+1, *ntypes)
		}

		for jtype = 1; jtype <= mtypes; jtype++ {
			if !(*dotype)[jtype-1] {
				goto label250
			}
			nmats = nmats + 1
			ntest = 0

			//           Save ISEED in case of an error.
			for j = 1; j <= 4; j++ {
				ioldsd[j-1] = (*iseed)[j-1]
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

			golapack.Zlaset('F', lda, &n, &czero, &czero, a, lda)
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
				matgen.Zlatmr(&n, &n, 'D', iseed, 'N', work, &imode, &cond, &cone, 'T', 'N', work.Off(n+1-1), func() *int { y := 1; return &y }(), &one, work.Off(2*n+1-1), func() *int { y := 1; return &y }(), &one, 'N', &idumma, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &zero, &anorm, 'N', a, lda, iwork, &iinfo)

			} else if itype == 5 {
				//              Hermitian, eigenvalues specified
				matgen.Zlatms(&n, &n, 'D', iseed, 'H', rwork, &imode, &cond, &anorm, &n, &n, 'N', a, lda, work, &iinfo)

			} else if itype == 6 {
				//              General, eigenvalues specified
				if kconds[jtype-1] == 1 {
					conds = one
				} else if kconds[jtype-1] == 2 {
					conds = rtulpi
				} else {
					conds = zero
				}

				matgen.Zlatme(&n, 'D', iseed, work, &imode, &cond, &cone, 'T', 'T', 'T', rwork, func() *int { y := 4; return &y }(), &conds, &n, &n, &anorm, a, lda, work.Off(n+1-1), &iinfo)

			} else if itype == 7 {
				//              Diagonal, random eigenvalues
				matgen.Zlatmr(&n, &n, 'D', iseed, 'N', work, func() *int { y := 6; return &y }(), &one, &cone, 'T', 'N', work.Off(n+1-1), func() *int { y := 1; return &y }(), &one, work.Off(2*n+1-1), func() *int { y := 1; return &y }(), &one, 'N', &idumma, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &zero, &anorm, 'N', a, lda, iwork, &iinfo)

			} else if itype == 8 {
				//              Hermitian, random eigenvalues
				matgen.Zlatmr(&n, &n, 'D', iseed, 'H', work, func() *int { y := 6; return &y }(), &one, &cone, 'T', 'N', work.Off(n+1-1), func() *int { y := 1; return &y }(), &one, work.Off(2*n+1-1), func() *int { y := 1; return &y }(), &one, 'N', &idumma, &n, &n, &zero, &anorm, 'N', a, lda, iwork, &iinfo)

			} else if itype == 9 {
				//              General, random eigenvalues
				matgen.Zlatmr(&n, &n, 'D', iseed, 'N', work, func() *int { y := 6; return &y }(), &one, &cone, 'T', 'N', work.Off(n+1-1), func() *int { y := 1; return &y }(), &one, work.Off(2*n+1-1), func() *int { y := 1; return &y }(), &one, 'N', &idumma, &n, &n, &zero, &anorm, 'N', a, lda, iwork, &iinfo)

			} else if itype == 10 {
				//              Triangular, random eigenvalues
				matgen.Zlatmr(&n, &n, 'D', iseed, 'N', work, func() *int { y := 6; return &y }(), &one, &cone, 'T', 'N', work.Off(n+1-1), func() *int { y := 1; return &y }(), &one, work.Off(2*n+1-1), func() *int { y := 1; return &y }(), &one, 'N', &idumma, &n, func() *int { y := 0; return &y }(), &zero, &anorm, 'N', a, lda, iwork, &iinfo)

			} else {

				iinfo = 1
			}

			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Generator", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				return
			}

		label100:
			;

			//           Call ZGEHRD to compute H and U, do tests.
			golapack.Zlacpy(' ', &n, &n, a, lda, h, lda)
			ntest = 1

			ilo = 1
			ihi = n

			golapack.Zgehrd(&n, &ilo, &ihi, h, lda, work, work.Off(n+1-1), toPtr((*nwork)-n), &iinfo)

			if iinfo != 0 {
				t.Fail()
				result.Set(0, ulpinv)
				fmt.Printf(" ZCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZGEHRD", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label240
			}

			for j = 1; j <= n-1; j++ {
				uu.Set(j+1-1, j-1, czero)
				for i = j + 2; i <= n; i++ {
					u.Set(i-1, j-1, h.Get(i-1, j-1))
					uu.Set(i-1, j-1, h.Get(i-1, j-1))
					h.Set(i-1, j-1, czero)
				}
			}
			goblas.Zcopy(toPtr(n-1), work, func() *int { y := 1; return &y }(), tau, func() *int { y := 1; return &y }())
			golapack.Zunghr(&n, &ilo, &ihi, u, ldu, work, work.Off(n+1-1), toPtr((*nwork)-n), &iinfo)
			ntest = 2

			Zhst01(&n, &ilo, &ihi, a, lda, h, lda, u, ldu, work, nwork, rwork, result.Off(0))

			//           Call ZHSEQR to compute T1, T2 and Z, do tests.
			//
			//           Eigenvalues only (W3)
			golapack.Zlacpy(' ', &n, &n, h, lda, t2, lda)
			ntest = 3
			result.Set(2, ulpinv)

			golapack.Zhseqr('E', 'N', &n, &ilo, &ihi, t2, lda, w3, uz, ldu, work, nwork, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZHSEQR(E)", iinfo, n, jtype, ioldsd)
				if iinfo <= n+2 {
					(*info) = absint(iinfo)
					goto label240
				}
			}

			//           Eigenvalues (W1) and Full Schur Form (T2)
			golapack.Zlacpy(' ', &n, &n, h, lda, t2, lda)

			golapack.Zhseqr('S', 'N', &n, &ilo, &ihi, t2, lda, w1, uz, ldu, work, nwork, &iinfo)
			if iinfo != 0 && iinfo <= n+2 {
				t.Fail()
				fmt.Printf(" ZCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZHSEQR(S)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label240
			}

			//           Eigenvalues (W1), Schur Form (T1), and Schur Vectors (UZ)
			golapack.Zlacpy(' ', &n, &n, h, lda, t1, lda)
			golapack.Zlacpy(' ', &n, &n, u, ldu, uz, ldu)

			golapack.Zhseqr('S', 'V', &n, &ilo, &ihi, t1, lda, w1, uz, ldu, work, nwork, &iinfo)
			if iinfo != 0 && iinfo <= n+2 {
				t.Fail()
				fmt.Printf(" ZCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZHSEQR(V)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label240
			}

			//           Compute Z = U' UZ
			goblas.Zgemm(ConjTrans, NoTrans, &n, &n, &n, &cone, u, ldu, uz, ldu, &czero, z, ldu)
			ntest = 8

			//           Do Tests 3: | H - Z T Z' | / ( |H| n ulp )
			//                and 4: | I - Z Z' | / ( n ulp )
			Zhst01(&n, &ilo, &ihi, h, lda, t1, lda, z, ldu, work, nwork, rwork, result.Off(2))

			//           Do Tests 5: | A - UZ T (UZ)' | / ( |A| n ulp )
			//                and 6: | I - UZ (UZ)' | / ( n ulp )
			Zhst01(&n, &ilo, &ihi, a, lda, t1, lda, uz, ldu, work, nwork, rwork, result.Off(4))

			//           Do Test 7: | T2 - T1 | / ( |T| n ulp )
			Zget10(&n, &n, t2, lda, t1, lda, work, rwork, result.GetPtr(6))

			//           Do Test 8: | W3 - W1 | / ( maxint(|W1|,|W3|) ulp )
			temp1 = zero
			temp2 = zero
			for j = 1; j <= n; j++ {
				temp1 = maxf64(temp1, w1.GetMag(j-1), w3.GetMag(j-1))
				temp2 = maxf64(temp2, cmplx.Abs(w1.Get(j-1)-w3.Get(j-1)))
			}

			result.Set(7, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

			//           Compute the Left and Right Eigenvectors of T
			//
			//           Compute the Right eigenvector Matrix:
			ntest = 9
			result.Set(8, ulpinv)

			//           _select every other eigenvector
			for j = 1; j <= n; j++ {
				(*_select)[j-1] = false
			}
			for j = 1; j <= n; j += 2 {
				(*_select)[j-1] = true
			}
			golapack.Ztrevc('R', 'A', *_select, &n, t1, lda, cdumma.CMatrix(*ldu, opts), ldu, evectr, ldu, &n, &in, work, rwork, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZTREVC(R,A)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label240
			}

			//           Test 9:  | TR - RW | / ( |T| |R| ulp )
			Zget22('N', 'N', 'N', &n, t1, lda, evectr, ldu, w1, work, rwork, dumma.Off(0))
			result.Set(8, dumma.Get(0))
			if dumma.Get(1) > (*thresh) {
				t.Fail()
				fmt.Printf(" ZCHKHS: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Right", "ZTREVC", dumma.Get(1), n, jtype, ioldsd)
			}

			//           Compute selected right eigenvectors and confirm that
			//           they agree with previous right eigenvectors
			golapack.Ztrevc('R', 'S', *_select, &n, t1, lda, cdumma.CMatrix(*ldu, opts), ldu, evectl, ldu, &n, &in, work, rwork, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZTREVC(R,S)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label240
			}

			k = 1
			match = true
			for j = 1; j <= n; j++ {
				if (*_select)[j-1] {
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
				fmt.Printf(" ZCHKHS: Selected %s Eigenvectors from %s do not match other eigenvectors          N=%6d, JTYPE=%6d, ISEED=%5d\n", "Right", "ZTREVC", n, jtype, ioldsd)
			}

			//           Compute the Left eigenvector Matrix:
			ntest = 10
			result.Set(9, ulpinv)
			golapack.Ztrevc('L', 'A', *_select, &n, t1, lda, evectl, ldu, cdumma.CMatrix(*ldu, opts), ldu, &n, &in, work, rwork, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZTREVC(L,A)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label240
			}

			//           Test 10:  | LT - WL | / ( |T| |L| ulp )
			Zget22('C', 'N', 'C', &n, t1, lda, evectl, ldu, w1, work, rwork, dumma.Off(2))
			result.Set(9, dumma.Get(2))
			if dumma.Get(3) > (*thresh) {
				t.Fail()
				fmt.Printf(" ZCHKHS: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Left", "ZTREVC", dumma.Get(3), n, jtype, ioldsd)
			}

			//           Compute selected left eigenvectors and confirm that
			//           they agree with previous left eigenvectors
			golapack.Ztrevc('L', 'S', *_select, &n, t1, lda, evectr, ldu, cdumma.CMatrix(*ldu, opts), ldu, &n, &in, work, rwork, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZTREVC(L,S)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label240
			}

			k = 1
			match = true
			for j = 1; j <= n; j++ {
				if (*_select)[j-1] {
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
				fmt.Printf(" ZCHKHS: Selected %s Eigenvectors from %s do not match other eigenvectors          N=%6d, JTYPE=%6d, ISEED=%5d\n", "Left", "ZTREVC", n, jtype, ioldsd)
			}

			//           Call ZHSEIN for Right eigenvectors of H, do test 11
			ntest = 11
			result.Set(10, ulpinv)
			for j = 1; j <= n; j++ {
				(*_select)[j-1] = true
			}

			golapack.Zhsein('R', 'Q', 'N', _select, &n, h, lda, w3, cdumma.CMatrix(*ldu, opts), ldu, evectx, ldu, &n1, &in, work, rwork, iwork, iwork, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZHSEIN(R)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				if iinfo < 0 {
					goto label240
				}
			} else {
				//              Test 11:  | HX - XW | / ( |H| |X| ulp )
				//
				//                        (from inverse iteration)
				Zget22('N', 'N', 'N', &n, h, lda, evectx, ldu, w3, work, rwork, dumma.Off(0))
				if dumma.Get(0) < ulpinv {
					result.Set(10, dumma.Get(0)*aninv)
				}
				if dumma.Get(1) > (*thresh) {
					t.Fail()
					fmt.Printf(" ZCHKHS: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Right", "ZHSEIN", dumma.Get(1), n, jtype, ioldsd)
				}
			}

			//           Call ZHSEIN for Left eigenvectors of H, do test 12
			ntest = 12
			result.Set(11, ulpinv)
			for j = 1; j <= n; j++ {
				(*_select)[j-1] = true
			}

			golapack.Zhsein('L', 'Q', 'N', _select, &n, h, lda, w3, evecty, ldu, cdumma.CMatrix(*ldu, opts), ldu, &n1, &in, work, rwork, iwork, iwork, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZHSEIN(L)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				if iinfo < 0 {
					goto label240
				}
			} else {
				//              Test 12:  | YH - WY | / ( |H| |Y| ulp )
				//
				//                        (from inverse iteration)
				Zget22('C', 'N', 'C', &n, h, lda, evecty, ldu, w3, work, rwork, dumma.Off(2))
				if dumma.Get(2) < ulpinv {
					result.Set(11, dumma.Get(2)*aninv)
				}
				if dumma.Get(3) > (*thresh) {
					t.Fail()
					fmt.Printf(" ZCHKHS: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Left", "ZHSEIN", dumma.Get(3), n, jtype, ioldsd)
				}
			}

			//           Call ZUNMHR for Right eigenvectors of A, do test 13
			ntest = 13
			result.Set(12, ulpinv)

			golapack.Zunmhr('L', 'N', &n, &n, &ilo, &ihi, uu, ldu, tau, evectx, ldu, work, nwork, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZUNMHR(L)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				if iinfo < 0 {
					goto label240
				}
			} else {
				//              Test 13:  | AX - XW | / ( |A| |X| ulp )
				//
				//                        (from inverse iteration)
				Zget22('N', 'N', 'N', &n, a, lda, evectx, ldu, w3, work, rwork, dumma.Off(0))
				if dumma.Get(0) < ulpinv {
					result.Set(12, dumma.Get(0)*aninv)
				}
			}

			//           Call ZUNMHR for Left eigenvectors of A, do test 14
			ntest = 14
			result.Set(13, ulpinv)

			golapack.Zunmhr('L', 'N', &n, &n, &ilo, &ihi, uu, ldu, tau, evecty, ldu, work, nwork, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "ZUNMHR(L)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				if iinfo < 0 {
					goto label240
				}
			} else {
				//              Test 14:  | YA - WY | / ( |A| |Y| ulp )
				//
				//                        (from inverse iteration)
				Zget22('C', 'N', 'C', &n, a, lda, evecty, ldu, w3, work, rwork, dumma.Off(2))
				if dumma.Get(2) < ulpinv {
					result.Set(13, dumma.Get(2)*aninv)
				}
			}

			//           End of Loop -- Check for RESULT(j) > THRESH
		label240:
			;

			ntestt = ntestt + ntest
			Dlafts([]byte("ZHS"), &n, &n, &jtype, &ntest, result, &ioldsd, thresh, &nerrs, t)

		label250:
		}
	label260:
	}

	//     Summary
	Dlasum([]byte("ZHS"), &nerrs, &ntestt)
}
