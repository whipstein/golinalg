package eig

import (
	"fmt"
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/golapack/gltest/matgen"
	"golinalg/mat"
	"math"
	"testing"
)

// Dchkhs checks the nonsymmetric eigenvalue problem routines.
//
//            DGEHRD factors A as  U H U' , where ' means transpose,
//            H is hessenberg, and U is an orthogonal matrix.
//
//            DORGHR generates the orthogonal matrix U.
//
//            DORMHR multiplies a matrix by the orthogonal matrix U.
//
//            DHSEQR factors H as  Z T Z' , where Z is orthogonal and
//            T is "quasi-triangular", and the eigenvalue vector W.
//
//            DTREVC computes the left and right eigenvector matrices
//            L and R for T.
//
//            DHSEIN computes the left and right eigenvector matrices
//            Y and X for H, using inverse iteration.
//
//    When DCHKHS is called, a number of matrix "sizes" ("n's") and a
//    number of matrix "types" are specified.  For each size ("n")
//    and each type of matrix, one matrix will be generated and used
//    to test the nonsymmetric eigenroutines.  For each matrix, 14
//    tests will be performed:
//
//    (1)     | A - U H U**T | / ( |A| n ulp )
//
//    (2)     | I - UU**T | / ( n ulp )
//
//    (3)     | H - Z T Z**T | / ( |H| n ulp )
//
//    (4)     | I - ZZ**T | / ( n ulp )
//
//    (5)     | A - UZ H (UZ)**T | / ( |A| n ulp )
//
//    (6)     | I - UZ (UZ)**T | / ( n ulp )
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
//    (7)  Same as (4), but multiplied by SQRT( overflow threshold )
//    (8)  Same as (4), but multiplied by SQRT( underflow threshold )
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
//    (17) Same as (16), but multiplied by SQRT( overflow threshold )
//    (18) Same as (16), but multiplied by SQRT( underflow threshold )
//
//    (19) Nonsymmetric matrix with random entries chosen from (-1,1).
//    (20) Same as (19), but multiplied by SQRT( overflow threshold )
//    (21) Same as (19), but multiplied by SQRT( underflow threshold )
func Dchkhs(nsizes *int, nn *[]int, ntypes *int, dotype *[]bool, iseed *[]int, thresh *float64, nounit *int, a *mat.Matrix, lda *int, h, t1, t2, u *mat.Matrix, ldu *int, z, uz *mat.Matrix, wr1, wi1, wr2, wi2, wr3, wi3 *mat.Vector, evectl, evectr, evecty, evectx, uu *mat.Matrix, tau, work *mat.Vector, nwork *int, iwork *[]int, _select *[]bool, result *mat.Vector, info *int, t *testing.T) {
	var badnn, match bool
	var aninv, anorm, cond, conds, one, ovfl, rtovfl, rtulp, rtulpi, rtunfl, temp1, temp2, ulp, ulpinv, unfl, zero float64
	var i, ihi, iinfo, ilo, imode, in, itype, j, jcol, jj, jsize, jtype, k, maxtyp, mtypes, n, n1, nerrs, nmats, nmax, nselc, nselr, ntest, ntestt int
	adumma := make([]byte, 1)
	idumma := make([]int, 1)
	ioldsd := make([]int, 4)
	kconds := make([]int, 21)
	kmagn := make([]int, 21)
	kmode := make([]int, 21)
	ktype := make([]int, 21)

	dumma := vf(6)

	zero = 0.0
	one = 1.0
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
		(*info) = -28
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DCHKHS"), -(*info))
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
			goto label270
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
				goto label260
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
			//       =5                 random log   symmetric, w/ eigenvalues
			//       =6                 random       general, w/ eigenvalues
			//       =7                              random diagonal
			//       =8                              random symmetric
			//       =9                              random general
			//       =10                             random triangular
			//
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

			golapack.Dlaset('F', lda, &n, &zero, &zero, a, lda)
			iinfo = 0
			cond = ulpinv

			//           Special Matrices
			if itype == 1 {
				//              Zero
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
				matgen.Dlatms(&n, &n, 'S', iseed, 'S', work, &imode, &cond, &anorm, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), 'N', a, lda, work.Off(n+1-1), &iinfo)

			} else if itype == 5 {
				//              Symmetric, eigenvalues specified
				matgen.Dlatms(&n, &n, 'S', iseed, 'S', work, &imode, &cond, &anorm, &n, &n, 'N', a, lda, work.Off(n+1-1), &iinfo)

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
				matgen.Dlatme(&n, 'S', iseed, work, &imode, &cond, &one, adumma, 'T', 'T', 'T', work.Off(n+1-1), func() *int { y := 4; return &y }(), &conds, &n, &n, &anorm, a, lda, work.Off(2*n+1-1), &iinfo)

			} else if itype == 7 {
				//              Diagonal, random eigenvalues
				matgen.Dlatmr(&n, &n, 'S', iseed, 'S', work, func() *int { y := 6; return &y }(), &one, &one, 'T', 'N', work.Off(n+1-1), func() *int { y := 1; return &y }(), &one, work.Off(2*n+1-1), func() *int { y := 1; return &y }(), &one, 'N', &idumma, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &zero, &anorm, 'N', a, lda, iwork, &iinfo)

			} else if itype == 8 {
				//              Symmetric, random eigenvalues
				matgen.Dlatmr(&n, &n, 'S', iseed, 'S', work, func() *int { y := 6; return &y }(), &one, &one, 'T', 'N', work.Off(n+1-1), func() *int { y := 1; return &y }(), &one, work.Off(2*n+1-1), func() *int { y := 1; return &y }(), &one, 'N', &idumma, &n, &n, &zero, &anorm, 'N', a, lda, iwork, &iinfo)

			} else if itype == 9 {
				//              General, random eigenvalues
				matgen.Dlatmr(&n, &n, 'S', iseed, 'N', work, func() *int { y := 6; return &y }(), &one, &one, 'T', 'N', work.Off(n+1-1), func() *int { y := 1; return &y }(), &one, work.Off(2*n+1-1), func() *int { y := 1; return &y }(), &one, 'N', &idumma, &n, &n, &zero, &anorm, 'N', a, lda, iwork, &iinfo)

			} else if itype == 10 {
				//              Triangular, random eigenvalues
				matgen.Dlatmr(&n, &n, 'S', iseed, 'N', work, func() *int { y := 6; return &y }(), &one, &one, 'T', 'N', work.Off(n+1-1), func() *int { y := 1; return &y }(), &one, work.Off(2*n+1-1), func() *int { y := 1; return &y }(), &one, 'N', &idumma, &n, func() *int { y := 0; return &y }(), &zero, &anorm, 'N', a, lda, iwork, &iinfo)

			} else {

				iinfo = 1
			}

			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" DCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Generator", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				return
			}

		label100:
			;

			//           Call DGEHRD to compute H and U, do tests.
			golapack.Dlacpy(' ', &n, &n, a, lda, h, lda)

			ntest = 1

			ilo = 1
			ihi = n

			golapack.Dgehrd(&n, &ilo, &ihi, h, lda, work, work.Off(n+1-1), toPtr((*nwork)-n), &iinfo)

			if iinfo != 0 {
				t.Fail()
				result.Set(0, ulpinv)
				fmt.Printf(" DCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DGEHRD", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label250
			}

			for j = 1; j <= n-1; j++ {
				uu.Set(j+1-1, j-1, zero)
				for i = j + 2; i <= n; i++ {
					u.Set(i-1, j-1, h.Get(i-1, j-1))
					uu.Set(i-1, j-1, h.Get(i-1, j-1))
					h.Set(i-1, j-1, zero)
				}
			}
			goblas.Dcopy(toPtr(n-1), work, toPtr(1), tau, toPtr(1))
			golapack.Dorghr(&n, &ilo, &ihi, u, ldu, work, work.Off(n+1-1), toPtr((*nwork)-n), &iinfo)
			ntest = 2

			Dhst01(&n, &ilo, &ihi, a, lda, h, lda, u, ldu, work, nwork, result)

			//           Call DHSEQR to compute T1, T2 and Z, do tests.
			//
			//           Eigenvalues only (WR3,WI3)
			golapack.Dlacpy(' ', &n, &n, h, lda, t2, lda)
			ntest = 3
			result.Set(2, ulpinv)

			golapack.Dhseqr('E', 'N', &n, &ilo, &ihi, t2, lda, wr3, wi3, uz, ldu, work, nwork, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" DCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DHSEQR(E)", iinfo, n, jtype, ioldsd)
				if iinfo <= n+2 {
					(*info) = absint(iinfo)
					goto label250
				}
			}

			//           Eigenvalues (WR2,WI2) and Full Schur Form (T2)
			golapack.Dlacpy(' ', &n, &n, h, lda, t2, lda)

			golapack.Dhseqr('S', 'N', &n, &ilo, &ihi, t2, lda, wr2, wi2, uz, ldu, work, nwork, &iinfo)
			if iinfo != 0 && iinfo <= n+2 {
				t.Fail()
				fmt.Printf(" DCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DHSEQR(S)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label250
			}

			//           Eigenvalues (WR1,WI1), Schur Form (T1), and Schur vectors
			//           (UZ)
			golapack.Dlacpy(' ', &n, &n, h, lda, t1, lda)
			golapack.Dlacpy(' ', &n, &n, u, ldu, uz, ldu)

			golapack.Dhseqr('S', 'V', &n, &ilo, &ihi, t1, lda, wr1, wi1, uz, ldu, work, nwork, &iinfo)
			if iinfo != 0 && iinfo <= n+2 {
				t.Fail()
				fmt.Printf(" DCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DHSEQR(V)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label250
			}

			//           Compute Z = U' UZ
			goblas.Dgemm(Trans, NoTrans, &n, &n, &n, &one, u, ldu, uz, ldu, &zero, z, ldu)
			ntest = 8

			//           Do Tests 3: | H - Z T Z' | / ( |H| n ulp )
			//                and 4: | I - Z Z' | / ( n ulp )
			Dhst01(&n, &ilo, &ihi, h, lda, t1, lda, z, ldu, work, nwork, result.Off(2))

			//           Do Tests 5: | A - UZ T (UZ)' | / ( |A| n ulp )
			//                and 6: | I - UZ (UZ)' | / ( n ulp )
			Dhst01(&n, &ilo, &ihi, a, lda, t1, lda, uz, ldu, work, nwork, result.Off(4))

			//           Do Test 7: | T2 - T1 | / ( |T| n ulp )
			Dget10(&n, &n, t2, lda, t1, lda, work, result.GetPtr(6))

			//           Do Test 8: | W2 - W1 | / ( maxf64(|W1|,|W2|) ulp )
			temp1 = zero
			temp2 = zero
			for j = 1; j <= n; j++ {
				temp1 = maxf64(temp1, math.Abs(wr1.Get(j-1))+math.Abs(wi1.Get(j-1)), math.Abs(wr2.Get(j-1))+math.Abs(wi2.Get(j-1)))
				temp2 = maxf64(temp2, math.Abs(wr1.Get(j-1)-wr2.Get(j-1))+math.Abs(wi1.Get(j-1)-wi2.Get(j-1)))
			}

			result.Set(7, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

			//           Compute the Left and Right Eigenvectors of T
			//
			//           Compute the Right eigenvector Matrix:
			ntest = 9
			result.Set(8, ulpinv)

			//           Select last maxf64(N/4,1) real, maxf64(N/4,1) complex eigenvectors
			nselc = 0
			nselr = 0
			j = n
		label140:
			;
			if wi1.Get(j-1) == zero {
				if nselr < maxint(n/4, 1) {
					nselr = nselr + 1
					(*_select)[j-1] = true
				} else {
					(*_select)[j-1] = false
				}
				j = j - 1
			} else {
				if nselc < maxint(n/4, 1) {
					nselc = nselc + 1
					(*_select)[j-1] = true
					(*_select)[j-1-1] = false
				} else {
					(*_select)[j-1] = false
					(*_select)[j-1-1] = false
				}
				j = j - 2
			}
			if j > 0 {
				goto label140
			}

			golapack.Dtrevc('R', 'A', _select, &n, t1, lda, dumma.Matrix(*ldu, opts), ldu, evectr, ldu, &n, &in, work, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" DCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DTREVC(R,A)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label250
			}

			//           Test 9:  | TR - RW | / ( |T| |R| ulp )
			Dget22('N', 'N', 'N', &n, t1, lda, evectr, ldu, wr1, wi1, work, dumma)
			result.Set(8, dumma.Get(0))
			if dumma.Get(1) > (*thresh) {
				t.Fail()
				fmt.Printf(" DCHKHS: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Right", "DTREVC", dumma.Get(1), n, jtype, ioldsd)
			}

			//           Compute selected right eigenvectors and confirm that
			//           they agree with previous right eigenvectors
			golapack.Dtrevc('R', 'S', _select, &n, t1, lda, dumma.Matrix(*ldu, opts), ldu, evectl, ldu, &n, &in, work, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" DCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DTREVC(R,S)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label250
			}

			k = 1
			match = true
			for j = 1; j <= n; j++ {
				if (*_select)[j-1] && wi1.Get(j-1) == zero {
					for jj = 1; jj <= n; jj++ {
						if evectr.Get(jj-1, j-1) != evectl.Get(jj-1, k-1) {
							match = false
							goto label180
						}
					}
					k = k + 1
				} else if (*_select)[j-1] && wi1.Get(j-1) != zero {
					for jj = 1; jj <= n; jj++ {
						if evectr.Get(jj-1, j-1) != evectl.Get(jj-1, k-1) || evectr.Get(jj-1, j+1-1) != evectl.Get(jj-1, k+1-1) {
							match = false
							goto label180
						}
					}
					k = k + 2
				}
			}
		label180:
			;
			if !match {
				t.Fail()
				fmt.Printf(" DCHKHS: Selected %s Eigenvectors from %s do not match other eigenvectors          N=%6d, JTYPE=%6d, ISEED=%5d\n", "Right", "DTREVC", n, jtype, ioldsd)
			}

			//           Compute the Left eigenvector Matrix:
			ntest = 10
			result.Set(9, ulpinv)
			golapack.Dtrevc('L', 'A', _select, &n, t1, lda, evectl, ldu, dumma.Matrix(*ldu, opts), ldu, &n, &in, work, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" DCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DTREVC(L,A)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label250
			}

			//           Test 10:  | LT - WL | / ( |T| |L| ulp )
			Dget22('T', 'N', 'C', &n, t1, lda, evectl, ldu, wr1, wi1, work, dumma.Off(2))
			result.Set(9, dumma.Get(2))
			if dumma.Get(3) > (*thresh) {
				t.Fail()
				fmt.Printf(" DCHKHS: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Left", "DTREVC", dumma.Get(3), n, jtype, ioldsd)
			}

			//           Compute selected left eigenvectors and confirm that
			//           they agree with previous left eigenvectors
			golapack.Dtrevc('L', 'S', _select, &n, t1, lda, evectr, ldu, dumma.Matrix(*ldu, opts), ldu, &n, &in, work, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" DCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DTREVC(L,S)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				goto label250
			}

			k = 1
			match = true
			for j = 1; j <= n; j++ {
				if (*_select)[j-1] && wi1.Get(j-1) == zero {
					for jj = 1; jj <= n; jj++ {
						if evectl.Get(jj-1, j-1) != evectr.Get(jj-1, k-1) {
							match = false
							goto label220
						}
					}
					k = k + 1
				} else if (*_select)[j-1] && wi1.Get(j-1) != zero {
					for jj = 1; jj <= n; jj++ {
						if evectl.Get(jj-1, j-1) != evectr.Get(jj-1, k-1) || evectl.Get(jj-1, j+1-1) != evectr.Get(jj-1, k+1-1) {
							match = false
							goto label220
						}
					}
					k = k + 2
				}
			}
		label220:
			;
			if !match {
				t.Fail()
				fmt.Printf(" DCHKHS: Selected %s Eigenvectors from %s do not match other eigenvectors          N=%6d, JTYPE=%6d, ISEED=%5d\n", "Left", "DTREVC", n, jtype, ioldsd)
			}

			//           Call DHSEIN for Right eigenvectors of H, do test 11
			ntest = 11
			result.Set(10, ulpinv)
			for j = 1; j <= n; j++ {
				(*_select)[j-1] = true
			}

			golapack.Dhsein('R', 'Q', 'N', _select, &n, h, lda, wr3, wi3, dumma.Matrix(*ldu, opts), ldu, evectx, ldu, &n1, &in, work, iwork, iwork, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" DCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DHSEIN(R)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				if iinfo < 0 {
					goto label250
				}
			} else {
				//              Test 11:  | HX - XW | / ( |H| |X| ulp )
				//
				//                        (from inverse iteration)
				Dget22('N', 'N', 'N', &n, h, lda, evectx, ldu, wr3, wi3, work, dumma)
				if dumma.Get(0) < ulpinv {
					result.Set(10, dumma.Get(0)*aninv)
				}
				if dumma.Get(1) > (*thresh) {
					t.Fail()
					fmt.Printf(" DCHKHS: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Right", "DHSEIN", dumma.Get(1), n, jtype, ioldsd)
				}
			}

			//           Call DHSEIN for Left eigenvectors of H, do test 12
			ntest = 12
			result.Set(11, ulpinv)
			for j = 1; j <= n; j++ {
				(*_select)[j-1] = true
			}

			golapack.Dhsein('L', 'Q', 'N', _select, &n, h, lda, wr3, wi3, evecty, ldu, dumma.Matrix(*ldu, opts), ldu, &n1, &in, work, iwork, iwork, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" DCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DHSEIN(L)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				if iinfo < 0 {
					goto label250
				}
			} else {
				//              Test 12:  | YH - WY | / ( |H| |Y| ulp )
				//
				//                        (from inverse iteration)
				Dget22('C', 'N', 'C', &n, h, lda, evecty, ldu, wr3, wi3, work, dumma.Off(2))
				if dumma.Get(2) < ulpinv {
					result.Set(11, dumma.Get(2)*aninv)
				}
				if dumma.Get(3) > (*thresh) {
					t.Fail()
					fmt.Printf(" DCHKHS: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Left", "DHSEIN", dumma.Get(3), n, jtype, ioldsd)
				}
			}

			//           Call DORMHR for Right eigenvectors of A, do test 13
			ntest = 13
			result.Set(12, ulpinv)

			golapack.Dormhr('L', 'N', &n, &n, &ilo, &ihi, uu, ldu, tau, evectx, ldu, work, nwork, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" DCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DORMHR(R)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				if iinfo < 0 {
					goto label250
				}
			} else {
				//              Test 13:  | AX - XW | / ( |A| |X| ulp )
				//
				//                        (from inverse iteration)
				Dget22('N', 'N', 'N', &n, a, lda, evectx, ldu, wr3, wi3, work, dumma)
				if dumma.Get(0) < ulpinv {
					result.Set(12, dumma.Get(0)*aninv)
				}
			}

			//           Call DORMHR for Left eigenvectors of A, do test 14
			ntest = 14
			result.Set(13, ulpinv)

			golapack.Dormhr('L', 'N', &n, &n, &ilo, &ihi, uu, ldu, tau, evecty, ldu, work, nwork, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" DCHKHS: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DORMHR(L)", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				if iinfo < 0 {
					goto label250
				}
			} else {
				//              Test 14:  | YA - WY | / ( |A| |Y| ulp )
				//
				//                        (from inverse iteration)
				Dget22('C', 'N', 'C', &n, a, lda, evecty, ldu, wr3, wi3, work, dumma.Off(2))
				if dumma.Get(2) < ulpinv {
					result.Set(13, dumma.Get(2)*aninv)
				}
			}

			//           End of Loop -- Check for RESULT(j) > THRESH
		label250:
			;

			ntestt = ntestt + ntest
			Dlafts([]byte("DHS"), &n, &n, &jtype, &ntest, result, &ioldsd, thresh, &nerrs, t)

		label260:
		}
	label270:
	}

	//     Summary
	Dlasum([]byte("DHS"), &nerrs, &ntestt)
}
