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

// Zdrvst2stg checks the Hermitian eigenvalue problem drivers.
//
//              ZHEEVD computes all eigenvalues and, optionally,
//              eigenvectors of a complex Hermitian matrix,
//              using a divide-and-conquer algorithm.
//
//              ZHEEVX computes selected eigenvalues and, optionally,
//              eigenvectors of a complex Hermitian matrix.
//
//              ZHEEVR computes selected eigenvalues and, optionally,
//              eigenvectors of a complex Hermitian matrix
//              using the Relatively Robust Representation where it can.
//
//              ZHPEVD computes all eigenvalues and, optionally,
//              eigenvectors of a complex Hermitian matrix in packed
//              storage, using a divide-and-conquer algorithm.
//
//              ZHPEVX computes selected eigenvalues and, optionally,
//              eigenvectors of a complex Hermitian matrix in packed
//              storage.
//
//              ZHBEVD computes all eigenvalues and, optionally,
//              eigenvectors of a complex Hermitian band matrix,
//              using a divide-and-conquer algorithm.
//
//              ZHBEVX computes selected eigenvalues and, optionally,
//              eigenvectors of a complex Hermitian band matrix.
//
//              ZHEEV computes all eigenvalues and, optionally,
//              eigenvectors of a complex Hermitian matrix.
//
//              ZHPEV computes all eigenvalues and, optionally,
//              eigenvectors of a complex Hermitian matrix in packed
//              storage.
//
//              ZHBEV computes all eigenvalues and, optionally,
//              eigenvectors of a complex Hermitian band matrix.
//
//      When ZDRVST2STG is called, a number of matrix "sizes" ("n's") and a
//      number of matrix "types" are specified.  For each size ("n")
//      and each _type of matrix, one matrix will be generated and used
//      to test the appropriate drivers.  For each matrix and each
//      driver routine called, the following tests will be performed:
//
//      (1)     | A - Z D Z' | / ( |A| n ulp )
//
//      (2)     | I - Z Z' | / ( n ulp )
//
//      (3)     | D1 - D2 | / ( |D1| ulp )
//
//      where Z is the matrix of eigenvectors returned when the
//      eigenvector option is given and D1 and D2 are the eigenvalues
//      returned with and without the eigenvector option.
//
//      The "sizes" are specified by an array NN(1:NSIZES); the value of
//      each element NN(j) specifies one size.
//      The "types" are specified by a logical array DOTYPE( 1:NTYPES );
//      if DOTYPE(j) is .TRUE., then matrix _type "j" will be generated.
//      Currently, the list of possible types is:
//
//      (1)  The zero matrix.
//      (2)  The identity matrix.
//
//      (3)  A diagonal matrix with evenly spaced entries
//           1, ..., ULP  and random signs.
//           (ULP = (first number larger than 1) - 1 )
//      (4)  A diagonal matrix with geometrically spaced entries
//           1, ..., ULP  and random signs.
//      (5)  A diagonal matrix with "clustered" entries 1, ULP, ..., ULP
//           and random signs.
//
//      (6)  Same as (4), but multiplied by SQRT( overflow threshold )
//      (7)  Same as (4), but multiplied by SQRT( underflow threshold )
//
//      (8)  A matrix of the form  U* D U, where U is unitary and
//           D has evenly spaced entries 1, ..., ULP with random signs
//           on the diagonal.
//
//      (9)  A matrix of the form  U* D U, where U is unitary and
//           D has geometrically spaced entries 1, ..., ULP with random
//           signs on the diagonal.
//
//      (10) A matrix of the form  U* D U, where U is unitary and
//           D has "clustered" entries 1, ULP,..., ULP with random
//           signs on the diagonal.
//
//      (11) Same as (8), but multiplied by SQRT( overflow threshold )
//      (12) Same as (8), but multiplied by SQRT( underflow threshold )
//
//      (13) Symmetric matrix with random entries chosen from (-1,1).
//      (14) Same as (13), but multiplied by SQRT( overflow threshold )
//      (15) Same as (13), but multiplied by SQRT( underflow threshold )
//      (16) A band matrix with half bandwidth randomly chosen between
//           0 and N-1, with evenly spaced eigenvalues 1, ..., ULP
//           with random signs.
//      (17) Same as (16), but multiplied by SQRT( overflow threshold )
//      (18) Same as (16), but multiplied by SQRT( underflow threshold )
func Zdrvst2stg(nsizes *int, nn *[]int, ntypes *int, dotype *[]bool, iseed *[]int, thresh *float64, nounit *int, a *mat.CMatrix, lda *int, d1, d2, d3, wa1, wa2, wa3 *mat.Vector, u *mat.CMatrix, ldu *int, v *mat.CMatrix, tau *mat.CVector, z *mat.CMatrix, work *mat.CVector, lwork *int, rwork *mat.Vector, lrwork *int, iwork *[]int, liwork *int, result *mat.Vector, info *int, t *testing.T) {
	var badnn bool
	var uplo byte
	var cone, czero complex128
	var abstol, aninv, anorm, cond, half, one, ovfl, rtovfl, rtunfl, temp1, temp2, temp3, ten, two, ulp, ulpinv, unfl, vl, vu, zero float64
	var i, idiag, ihbw, iinfo, il, imode, indwrk, indx, irow, itemp, itype, iu, iuplo, j, j1, j2, jcol, jsize, jtype, kd, lgn, liwedc, lrwedc, lwedc, m, m2, m3, maxtyp, mtypes, n, nerrs, nmats, nmax, ntest, ntestt int
	idumma := make([]int, 1)
	ioldsd := make([]int, 4)
	iseed2 := make([]int, 4)
	iseed3 := make([]int, 4)
	kmagn := make([]int, 18)
	kmode := make([]int, 18)
	ktype := make([]int, 18)

	zero = 0.0
	one = 1.0
	two = 2.0
	ten = 10.0
	half = one / two
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	maxtyp = 18

	ktype[0], ktype[1], ktype[2], ktype[3], ktype[4], ktype[5], ktype[6], ktype[7], ktype[8], ktype[9], ktype[10], ktype[11], ktype[12], ktype[13], ktype[14], ktype[15], ktype[16], ktype[17] = 1, 2, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 8, 8, 8, 9, 9, 9
	kmagn[0], kmagn[1], kmagn[2], kmagn[3], kmagn[4], kmagn[5], kmagn[6], kmagn[7], kmagn[8], kmagn[9], kmagn[10], kmagn[11], kmagn[12], kmagn[13], kmagn[14], kmagn[15], kmagn[16], kmagn[17] = 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3
	kmode[0], kmode[1], kmode[2], kmode[3], kmode[4], kmode[5], kmode[6], kmode[7], kmode[8], kmode[9], kmode[10], kmode[11], kmode[12], kmode[13], kmode[14], kmode[15], kmode[16], kmode[17] = 0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0, 4, 4, 4

	//     1)      Check for errors
	ntestt = 0
	(*info) = 0

	badnn = false
	nmax = 1
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
	} else if (*lda) < nmax {
		(*info) = -9
	} else if (*ldu) < nmax {
		(*info) = -16
	} else if 2*powint(maxint(2, nmax), 2) > (*lwork) {
		(*info) = -22
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZDRVST2STG"), -(*info))
		return
	}

	//     Quick return if nothing to do
	if (*nsizes) == 0 || (*ntypes) == 0 {
		return
	}

	//     More Important constants
	unfl = golapack.Dlamch(SafeMinimum)
	ovfl = golapack.Dlamch(Overflow)
	golapack.Dlabad(&unfl, &ovfl)
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)
	ulpinv = one / ulp
	rtunfl = math.Sqrt(unfl)
	rtovfl = math.Sqrt(ovfl)

	//     Loop over sizes, types
	for i = 1; i <= 4; i++ {
		iseed2[i-1] = (*iseed)[i-1]
		iseed3[i-1] = (*iseed)[i-1]
	}

	nerrs = 0
	nmats = 0

	for jsize = 1; jsize <= (*nsizes); jsize++ {
		n = (*nn)[jsize-1]
		if n > 0 {
			lgn = int(math.Log(float64(n)) / math.Log(two))
			if powint(2, lgn) < n {
				lgn = lgn + 1
			}
			if powint(2, lgn) < n {
				lgn = lgn + 1
			}
			lwedc = maxint(2*n+n*n, 2*n*n)
			lrwedc = 1 + 4*n + 2*n*lgn + 3*powint(n, 2)
			liwedc = 3 + 5*n
		} else {
			lwedc = 2
			lrwedc = 8
			liwedc = 8
		}
		aninv = one / float64(maxint(1, n))

		if (*nsizes) != 1 {
			mtypes = minint(maxtyp, *ntypes)
		} else {
			mtypes = minint(maxtyp+1, *ntypes)
		}

		for jtype = 1; jtype <= mtypes; jtype++ {
			if !(*dotype)[jtype-1] {
				goto label1210
			}
			nmats = nmats + 1
			ntest = 0

			for j = 1; j <= 4; j++ {
				ioldsd[j-1] = (*iseed)[j-1]
			}

			//           2)      Compute "A"
			//
			//                   Control parameters:
			//
			//               KMAGN  KMODE        KTYPE
			//           =1  O(1)   clustered 1  zero
			//           =2  large  clustered 2  identity
			//           =3  small  exponential  (none)
			//           =4         arithmetic   diagonal, (w/ eigenvalues)
			//           =5         random log   Hermitian, w/ eigenvalues
			//           =6         random       (none)
			//           =7                      random diagonal
			//           =8                      random Hermitian
			//           =9                      band Hermitian, w/ eigenvalues
			if mtypes > maxtyp {
				goto label110
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

			//           Special Matrices -- Identity & Jordan block
			//
			//                   Zero
			if itype == 1 {
				iinfo = 0

			} else if itype == 2 {
				//              Identity
				for jcol = 1; jcol <= n; jcol++ {
					a.SetRe(jcol-1, jcol-1, anorm)
				}

			} else if itype == 4 {
				//              Diagonal Matrix, [Eigen]values Specified
				matgen.Zlatms(&n, &n, 'S', iseed, 'H', rwork, &imode, &cond, &anorm, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), 'N', a, lda, work, &iinfo)

			} else if itype == 5 {
				//              Hermitian, eigenvalues specified
				matgen.Zlatms(&n, &n, 'S', iseed, 'H', rwork, &imode, &cond, &anorm, &n, &n, 'N', a, lda, work, &iinfo)

			} else if itype == 7 {
				//              Diagonal, random eigenvalues
				matgen.Zlatmr(&n, &n, 'S', iseed, 'H', work, func() *int { y := 6; return &y }(), &one, &cone, 'T', 'N', work.Off(n+1-1), func() *int { y := 1; return &y }(), &one, work.Off(2*n+1-1), func() *int { y := 1; return &y }(), &one, 'N', &idumma, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &zero, &anorm, 'N', a, lda, iwork, &iinfo)

			} else if itype == 8 {
				//              Hermitian, random eigenvalues
				matgen.Zlatmr(&n, &n, 'S', iseed, 'H', work, func() *int { y := 6; return &y }(), &one, &cone, 'T', 'N', work.Off(n+1-1), func() *int { y := 1; return &y }(), &one, work.Off(2*n+1-1), func() *int { y := 1; return &y }(), &one, 'N', &idumma, &n, &n, &zero, &anorm, 'N', a, lda, iwork, &iinfo)

			} else if itype == 9 {
				//              Hermitian banded, eigenvalues specified
				ihbw = int(float64(n-1) * matgen.Dlarnd(func() *int { y := 1; return &y }(), &iseed3))
				matgen.Zlatms(&n, &n, 'S', iseed, 'H', rwork, &imode, &cond, &anorm, &ihbw, &ihbw, 'Z', u, ldu, work, &iinfo)

				//              Store as dense matrix for most routines.
				golapack.Zlaset('F', lda, &n, &czero, &czero, a, lda)
				for idiag = -ihbw; idiag <= ihbw; idiag++ {
					irow = ihbw - idiag + 1
					j1 = maxint(1, idiag+1)
					j2 = minint(n, n+idiag)
					for j = j1; j <= j2; j++ {
						i = j - idiag
						a.Set(i-1, j-1, u.Get(irow-1, j-1))
					}
				}
			} else {
				iinfo = 1
			}

			if iinfo != 0 {
				fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Generator", iinfo, n, jtype, ioldsd)
				(*info) = absint(iinfo)
				return
			}

		label110:
			;

			abstol = unfl + unfl
			if n <= 1 {
				il = 1
				iu = n
			} else {
				il = 1 + int(float64(n-1)*matgen.Dlarnd(func() *int { y := 1; return &y }(), &iseed2))
				iu = 1 + int(float64(n-1)*matgen.Dlarnd(func() *int { y := 1; return &y }(), &iseed2))
				if il > iu {
					itemp = il
					il = iu
					iu = itemp
				}
			}

			//           Perform tests storing upper or lower triangular
			//           part of matrix.
			for iuplo = 0; iuplo <= 1; iuplo++ {
				if iuplo == 0 {
					uplo = 'L'
				} else {
					uplo = 'U'
				}

				//              Call ZHEEVD and CHEEVX.
				golapack.Zlacpy(' ', &n, &n, a, lda, v, ldu)
				//
				ntest = ntest + 1
				golapack.Zheevd('V', uplo, &n, a, ldu, d1, work, &lwedc, rwork, &lrwedc, iwork, &liwedc, &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHEEVD(V,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label130
					}
				}

				//              Do tests 1 and 2.
				Zhet21(func() *int { y := 1; return &y }(), uplo, &n, func() *int { y := 0; return &y }(), v, ldu, d1, d2, a, ldu, z, ldu, tau, work, rwork, result.Off(ntest-1))

				golapack.Zlacpy(' ', &n, &n, v, ldu, a, lda)

				ntest = ntest + 2
				golapack.Zheevd2stage('N', uplo, &n, a, ldu, d3, work, lwork, rwork, &lrwedc, iwork, &liwedc, &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHEEVD_2STAGE(N,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label130
					}
				}

				//              Do test 3.
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, d1.GetMag(j-1), d3.GetMag(j-1))
					temp2 = maxf64(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

			label130:
				;
				golapack.Zlacpy(' ', &n, &n, v, ldu, a, lda)

				ntest = ntest + 1

				if n > 0 {
					temp3 = maxf64(d1.GetMag(0), d1.GetMag(n-1))
					if il != 1 {
						vl = d1.Get(il-1) - maxf64(half*(d1.Get(il-1)-d1.Get(il-1-1)), ten*ulp*temp3, ten*rtunfl)
					} else if n > 0 {
						vl = d1.Get(0) - maxf64(half*(d1.Get(n-1)-d1.Get(0)), ten*ulp*temp3, ten*rtunfl)
					}
					if iu != n {
						vu = d1.Get(iu-1) + maxf64(half*(d1.Get(iu+1-1)-d1.Get(iu-1)), ten*ulp*temp3, ten*rtunfl)
					} else if n > 0 {
						vu = d1.Get(n-1) + maxf64(half*(d1.Get(n-1)-d1.Get(0)), ten*ulp*temp3, ten*rtunfl)
					}
				} else {
					temp3 = zero
					vl = zero
					vu = one
				}

				golapack.Zheevx('V', 'A', uplo, &n, a, ldu, &vl, &vu, &il, &iu, &abstol, &m, wa1, z, ldu, work, lwork, rwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHEEVX(V,A,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label150
					}
				}

				//              Do tests 4 and 5.
				golapack.Zlacpy(' ', &n, &n, v, ldu, a, lda)

				Zhet21(func() *int { y := 1; return &y }(), uplo, &n, func() *int { y := 0; return &y }(), a, ldu, wa1, d2, z, ldu, v, ldu, tau, work, rwork, result.Off(ntest-1))

				ntest = ntest + 2
				golapack.Zheevx2stage('N', 'A', uplo, &n, a, ldu, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, work, lwork, rwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHEEVX_2STAGE(N,A,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label150
					}
				}

				//              Do test 6.
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, wa1.GetMag(j-1), wa2.GetMag(j-1))
					temp2 = maxf64(temp2, math.Abs(wa1.Get(j-1)-wa2.Get(j-1)))
				}
				result.Set(ntest-1, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

			label150:
				;
				golapack.Zlacpy(' ', &n, &n, v, ldu, a, lda)

				ntest = ntest + 1

				golapack.Zheevx('V', 'I', uplo, &n, a, ldu, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, work, lwork, rwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHEEVX(V,I,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label160
					}
				}

				//              Do tests 7 and 8.
				golapack.Zlacpy(' ', &n, &n, v, ldu, a, lda)

				Zhet22(func() *int { y := 1; return &y }(), uplo, &n, &m2, func() *int { y := 0; return &y }(), a, ldu, wa2, d2, z, ldu, v, ldu, tau, work, rwork, result.Off(ntest-1))

				ntest = ntest + 2

				golapack.Zheevx2stage('N', 'I', uplo, &n, a, ldu, &vl, &vu, &il, &iu, &abstol, &m3, wa3, z, ldu, work, lwork, rwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHEEVX_2STAGE(N,I,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label160
					}
				}

				//              Do test 9.
				temp1 = Dsxt1(func() *int { y := 1; return &y }(), wa2, &m2, wa3, &m3, &abstol, &ulp, &unfl)
				temp2 = Dsxt1(func() *int { y := 1; return &y }(), wa3, &m3, wa2, &m2, &abstol, &ulp, &unfl)
				if n > 0 {
					temp3 = maxf64(wa1.GetMag(0), wa1.GetMag(n-1))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/maxf64(unfl, temp3*ulp))

			label160:
				;
				golapack.Zlacpy(' ', &n, &n, v, ldu, a, lda)

				ntest = ntest + 1

				golapack.Zheevx('V', 'V', uplo, &n, a, ldu, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, work, lwork, rwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHEEVX(V,V,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label170
					}
				}

				//              Do tests 10 and 11.
				golapack.Zlacpy(' ', &n, &n, v, ldu, a, lda)

				Zhet22(func() *int { y := 1; return &y }(), uplo, &n, &m2, func() *int { y := 0; return &y }(), a, ldu, wa2, d2, z, ldu, v, ldu, tau, work, rwork, result.Off(ntest-1))

				ntest = ntest + 2

				golapack.Zheevx2stage('N', 'V', uplo, &n, a, ldu, &vl, &vu, &il, &iu, &abstol, &m3, wa3, z, ldu, work, lwork, rwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHEEVX_2STAGE(N,V,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label170
					}
				}

				if m3 == 0 && n > 0 {
					result.Set(ntest-1, ulpinv)
					goto label170
				}

				//              Do test 12.
				temp1 = Dsxt1(func() *int { y := 1; return &y }(), wa2, &m2, wa3, &m3, &abstol, &ulp, &unfl)
				temp2 = Dsxt1(func() *int { y := 1; return &y }(), wa3, &m3, wa2, &m2, &abstol, &ulp, &unfl)
				if n > 0 {
					temp3 = maxf64(wa1.GetMag(0), wa1.GetMag(n-1))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/maxf64(unfl, temp3*ulp))

			label170:
				;

				//              Call ZHPEVD and CHPEVX.
				golapack.Zlacpy(' ', &n, &n, v, ldu, a, lda)

				//              Load array WORK with the upper or lower triangular
				//              part of the matrix in packed form.
				if iuplo == 1 {
					indx = 1
					for j = 1; j <= n; j++ {
						for i = 1; i <= j; i++ {
							work.Set(indx-1, a.Get(i-1, j-1))
							indx = indx + 1
						}
					}
				} else {
					indx = 1
					for j = 1; j <= n; j++ {
						for i = j; i <= n; i++ {
							work.Set(indx-1, a.Get(i-1, j-1))
							indx = indx + 1
						}
					}
				}

				ntest = ntest + 1
				indwrk = n*(n+1)/2 + 1
				golapack.Zhpevd('V', uplo, &n, work, d1, z, ldu, work.Off(indwrk-1), &lwedc, rwork, &lrwedc, iwork, &liwedc, &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHPEVD(V,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label270
					}
				}

				//              Do tests 13 and 14.
				Zhet21(func() *int { y := 1; return &y }(), uplo, &n, func() *int { y := 0; return &y }(), a, lda, d1, d2, z, ldu, v, ldu, tau, work, rwork, result.Off(ntest-1))

				if iuplo == 1 {
					indx = 1
					for j = 1; j <= n; j++ {
						for i = 1; i <= j; i++ {
							work.Set(indx-1, a.Get(i-1, j-1))
							indx = indx + 1
						}
					}
				} else {
					indx = 1
					for j = 1; j <= n; j++ {
						for i = j; i <= n; i++ {
							work.Set(indx-1, a.Get(i-1, j-1))
							indx = indx + 1
						}
					}
				}

				ntest = ntest + 2
				indwrk = n*(n+1)/2 + 1
				golapack.Zhpevd('N', uplo, &n, work, d3, z, ldu, work.Off(indwrk-1), &lwedc, rwork, &lrwedc, iwork, &liwedc, &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHPEVD(N,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label270
					}
				}

				//              Do test 15.
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, d1.GetMag(j-1), d3.GetMag(j-1))
					temp2 = maxf64(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

				//              Load array WORK with the upper or lower triangular part
				//              of the matrix in packed form.
			label270:
				;
				if iuplo == 1 {
					indx = 1
					for j = 1; j <= n; j++ {
						for i = 1; i <= j; i++ {
							work.Set(indx-1, a.Get(i-1, j-1))
							indx = indx + 1
						}
					}
				} else {
					indx = 1
					for j = 1; j <= n; j++ {
						for i = j; i <= n; i++ {
							work.Set(indx-1, a.Get(i-1, j-1))
							indx = indx + 1
						}
					}
				}

				ntest = ntest + 1

				if n > 0 {
					temp3 = maxf64(d1.GetMag(0), d1.GetMag(n-1))
					if il != 1 {
						vl = d1.Get(il-1) - maxf64(half*(d1.Get(il-1)-d1.Get(il-1-1)), ten*ulp*temp3, ten*rtunfl)
					} else if n > 0 {
						vl = d1.Get(0) - maxf64(half*(d1.Get(n-1)-d1.Get(0)), ten*ulp*temp3, ten*rtunfl)
					}
					if iu != n {
						vu = d1.Get(iu-1) + maxf64(half*(d1.Get(iu+1-1)-d1.Get(iu-1)), ten*ulp*temp3, ten*rtunfl)
					} else if n > 0 {
						vu = d1.Get(n-1) + maxf64(half*(d1.Get(n-1)-d1.Get(0)), ten*ulp*temp3, ten*rtunfl)
					}
				} else {
					temp3 = zero
					vl = zero
					vu = one
				}

				golapack.Zhpevx('V', 'A', uplo, &n, work, &vl, &vu, &il, &iu, &abstol, &m, wa1, z, ldu, v.CVector(0, 0), rwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHPEVX(V,A,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label370
					}
				}

				//              Do tests 16 and 17.
				Zhet21(func() *int { y := 1; return &y }(), uplo, &n, func() *int { y := 0; return &y }(), a, ldu, wa1, d2, z, ldu, v, ldu, tau, work, rwork, result.Off(ntest-1))

				ntest = ntest + 2

				if iuplo == 1 {
					indx = 1
					for j = 1; j <= n; j++ {
						for i = 1; i <= j; i++ {
							work.Set(indx-1, a.Get(i-1, j-1))
							indx = indx + 1
						}
					}
				} else {
					indx = 1
					for j = 1; j <= n; j++ {
						for i = j; i <= n; i++ {
							work.Set(indx-1, a.Get(i-1, j-1))
							indx = indx + 1
						}
					}
				}

				golapack.Zhpevx('N', 'A', uplo, &n, work, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, v.CVector(0, 0), rwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHPEVX(N,A,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label370
					}
				}

				//              Do test 18.
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, wa1.GetMag(j-1), wa2.GetMag(j-1))
					temp2 = maxf64(temp2, math.Abs(wa1.Get(j-1)-wa2.Get(j-1)))
				}
				result.Set(ntest-1, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

			label370:
				;
				ntest = ntest + 1
				if iuplo == 1 {
					indx = 1
					for j = 1; j <= n; j++ {
						for i = 1; i <= j; i++ {
							work.Set(indx-1, a.Get(i-1, j-1))
							indx = indx + 1
						}
					}
				} else {
					indx = 1
					for j = 1; j <= n; j++ {
						for i = j; i <= n; i++ {
							work.Set(indx-1, a.Get(i-1, j-1))
							indx = indx + 1
						}
					}
				}

				golapack.Zhpevx('V', 'I', uplo, &n, work, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, v.CVector(0, 0), rwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHPEVX(V,I,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label460
					}
				}

				//              Do tests 19 and 20.
				Zhet22(func() *int { y := 1; return &y }(), uplo, &n, &m2, func() *int { y := 0; return &y }(), a, ldu, wa2, d2, z, ldu, v, ldu, tau, work, rwork, result.Off(ntest-1))

				ntest = ntest + 2

				if iuplo == 1 {
					indx = 1
					for j = 1; j <= n; j++ {
						for i = 1; i <= j; i++ {
							work.Set(indx-1, a.Get(i-1, j-1))
							indx = indx + 1
						}
					}
				} else {
					indx = 1
					for j = 1; j <= n; j++ {
						for i = j; i <= n; i++ {
							work.Set(indx-1, a.Get(i-1, j-1))
							indx = indx + 1
						}
					}
				}

				golapack.Zhpevx('N', 'I', uplo, &n, work, &vl, &vu, &il, &iu, &abstol, &m3, wa3, z, ldu, v.CVector(0, 0), rwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHPEVX(N,I,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label460
					}
				}

				//              Do test 21.
				temp1 = Dsxt1(func() *int { y := 1; return &y }(), wa2, &m2, wa3, &m3, &abstol, &ulp, &unfl)
				temp2 = Dsxt1(func() *int { y := 1; return &y }(), wa3, &m3, wa2, &m2, &abstol, &ulp, &unfl)
				if n > 0 {
					temp3 = maxf64(wa1.GetMag(0), wa1.GetMag(n-1))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/maxf64(unfl, temp3*ulp))

			label460:
				;
				ntest = ntest + 1
				if iuplo == 1 {
					indx = 1
					for j = 1; j <= n; j++ {
						for i = 1; i <= j; i++ {
							work.Set(indx-1, a.Get(i-1, j-1))
							indx = indx + 1
						}
					}
				} else {
					indx = 1
					for j = 1; j <= n; j++ {
						for i = j; i <= n; i++ {
							work.Set(indx-1, a.Get(i-1, j-1))
							indx = indx + 1
						}
					}
				}

				golapack.Zhpevx('V', 'V', uplo, &n, work, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, v.CVector(0, 0), rwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHPEVX(V,V,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label550
					}
				}

				//              Do tests 22 and 23.
				Zhet22(func() *int { y := 1; return &y }(), uplo, &n, &m2, func() *int { y := 0; return &y }(), a, ldu, wa2, d2, z, ldu, v, ldu, tau, work, rwork, result.Off(ntest-1))

				ntest = ntest + 2

				if iuplo == 1 {
					indx = 1
					for j = 1; j <= n; j++ {
						for i = 1; i <= j; i++ {
							work.Set(indx-1, a.Get(i-1, j-1))
							indx = indx + 1
						}
					}
				} else {
					indx = 1
					for j = 1; j <= n; j++ {
						for i = j; i <= n; i++ {
							work.Set(indx-1, a.Get(i-1, j-1))
							indx = indx + 1
						}
					}
				}

				golapack.Zhpevx('N', 'V', uplo, &n, work, &vl, &vu, &il, &iu, &abstol, &m3, wa3, z, ldu, v.CVector(0, 0), rwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHPEVX(N,V,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label550
					}
				}

				if m3 == 0 && n > 0 {
					result.Set(ntest-1, ulpinv)
					goto label550
				}

				//              Do test 24.
				temp1 = Dsxt1(func() *int { y := 1; return &y }(), wa2, &m2, wa3, &m3, &abstol, &ulp, &unfl)
				temp2 = Dsxt1(func() *int { y := 1; return &y }(), wa3, &m3, wa2, &m2, &abstol, &ulp, &unfl)
				if n > 0 {
					temp3 = maxf64(wa1.GetMag(0), wa1.GetMag(n-1))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/maxf64(unfl, temp3*ulp))

			label550:
				;

				//              Call ZHBEVD and CHBEVX.
				if jtype <= 7 {
					kd = 0
				} else if jtype >= 8 && jtype <= 15 {
					kd = maxint(n-1, 0)
				} else {
					kd = ihbw
				}

				//              Load array V with the upper or lower triangular part
				//              of the matrix in band form.
				if iuplo == 1 {
					for j = 1; j <= n; j++ {
						for i = maxint(1, j-kd); i <= j; i++ {
							v.Set(kd+1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				} else {
					for j = 1; j <= n; j++ {
						for i = j; i <= minint(n, j+kd); i++ {
							v.Set(1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				}

				ntest = ntest + 1
				golapack.Zhbevd('V', uplo, &n, &kd, v, ldu, d1, z, ldu, work, &lwedc, rwork, &lrwedc, iwork, &liwedc, &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, KD=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHBEVD(V,"), uplo, ')'), iinfo, n, kd, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label650
					}
				}

				//              Do tests 25 and 26.
				Zhet21(func() *int { y := 1; return &y }(), uplo, &n, func() *int { y := 0; return &y }(), a, lda, d1, d2, z, ldu, v, ldu, tau, work, rwork, result.Off(ntest-1))

				if iuplo == 1 {
					for j = 1; j <= n; j++ {
						for i = maxint(1, j-kd); i <= j; i++ {
							v.Set(kd+1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				} else {
					for j = 1; j <= n; j++ {
						for i = j; i <= minint(n, j+kd); i++ {
							v.Set(1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				}

				ntest = ntest + 2
				golapack.Zhbevd2stage('N', uplo, &n, &kd, v, ldu, d3, z, ldu, work, lwork, rwork, &lrwedc, iwork, &liwedc, &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, KD=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHBEVD_2STAGE(N,"), uplo, ')'), iinfo, n, kd, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label650
					}
				}

				//              Do test 27.
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, d1.GetMag(j-1), d3.GetMag(j-1))
					temp2 = maxf64(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

				//              Load array V with the upper or lower triangular part
				//              of the matrix in band form.
			label650:
				;
				if iuplo == 1 {
					for j = 1; j <= n; j++ {
						for i = maxint(1, j-kd); i <= j; i++ {
							v.Set(kd+1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				} else {
					for j = 1; j <= n; j++ {
						for i = j; i <= minint(n, j+kd); i++ {
							v.Set(1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				}

				ntest = ntest + 1
				golapack.Zhbevx('V', 'A', uplo, &n, &kd, v, ldu, u, ldu, &vl, &vu, &il, &iu, &abstol, &m, wa1, z, ldu, work, rwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHBEVX(V,A,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label750
					}
				}

				//              Do tests 28 and 29.
				Zhet21(func() *int { y := 1; return &y }(), uplo, &n, func() *int { y := 0; return &y }(), a, ldu, wa1, d2, z, ldu, v, ldu, tau, work, rwork, result.Off(ntest-1))

				ntest = ntest + 2

				if iuplo == 1 {
					for j = 1; j <= n; j++ {
						for i = maxint(1, j-kd); i <= j; i++ {
							v.Set(kd+1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				} else {
					for j = 1; j <= n; j++ {
						for i = j; i <= minint(n, j+kd); i++ {
							v.Set(1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				}

				golapack.Zhbevx2stage('N', 'A', uplo, &n, &kd, v, ldu, u, ldu, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, work, lwork, rwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, KD=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHBEVX_2STAGE(N,A,"), uplo, ')'), iinfo, n, kd, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label750
					}
				}

				//              Do test 30.
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, wa1.GetMag(j-1), wa2.GetMag(j-1))
					temp2 = maxf64(temp2, math.Abs(wa1.Get(j-1)-wa2.Get(j-1)))
				}
				result.Set(ntest-1, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

				//              Load array V with the upper or lower triangular part
				//              of the matrix in band form.
			label750:
				;
				ntest = ntest + 1
				if iuplo == 1 {
					for j = 1; j <= n; j++ {
						for i = maxint(1, j-kd); i <= j; i++ {
							v.Set(kd+1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				} else {
					for j = 1; j <= n; j++ {
						for i = j; i <= minint(n, j+kd); i++ {
							v.Set(1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				}

				golapack.Zhbevx('V', 'I', uplo, &n, &kd, v, ldu, u, ldu, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, work, rwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, KD=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHBEVX(V,I,"), uplo, ')'), iinfo, n, kd, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label840
					}
				}

				//              Do tests 31 and 32.
				Zhet22(func() *int { y := 1; return &y }(), uplo, &n, &m2, func() *int { y := 0; return &y }(), a, ldu, wa2, d2, z, ldu, v, ldu, tau, work, rwork, result.Off(ntest-1))

				ntest = ntest + 2

				if iuplo == 1 {
					for j = 1; j <= n; j++ {
						for i = maxint(1, j-kd); i <= j; i++ {
							v.Set(kd+1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				} else {
					for j = 1; j <= n; j++ {
						for i = j; i <= minint(n, j+kd); i++ {
							v.Set(1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				}
				golapack.Zhbevx2stage('N', 'I', uplo, &n, &kd, v, ldu, u, ldu, &vl, &vu, &il, &iu, &abstol, &m3, wa3, z, ldu, work, lwork, rwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, KD=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHBEVX_2STAGE(N,I,"), uplo, ')'), iinfo, n, kd, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label840
					}
				}

				//              Do test 33.
				temp1 = Dsxt1(func() *int { y := 1; return &y }(), wa2, &m2, wa3, &m3, &abstol, &ulp, &unfl)
				temp2 = Dsxt1(func() *int { y := 1; return &y }(), wa3, &m3, wa2, &m2, &abstol, &ulp, &unfl)
				if n > 0 {
					temp3 = maxf64(wa1.GetMag(0), wa1.GetMag(n-1))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/maxf64(unfl, temp3*ulp))

				//              Load array V with the upper or lower triangular part
				//              of the matrix in band form.
			label840:
				;
				ntest = ntest + 1
				if iuplo == 1 {
					for j = 1; j <= n; j++ {
						for i = maxint(1, j-kd); i <= j; i++ {
							v.Set(kd+1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				} else {
					for j = 1; j <= n; j++ {
						for i = j; i <= minint(n, j+kd); i++ {
							v.Set(1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				}
				golapack.Zhbevx('V', 'V', uplo, &n, &kd, v, ldu, u, ldu, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, work, rwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, KD=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHBEVX(V,V,"), uplo, ')'), iinfo, n, kd, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label930
					}
				}

				//              Do tests 34 and 35.
				Zhet22(func() *int { y := 1; return &y }(), uplo, &n, &m2, func() *int { y := 0; return &y }(), a, ldu, wa2, d2, z, ldu, v, ldu, tau, work, rwork, result.Off(ntest-1))

				ntest = ntest + 2

				if iuplo == 1 {
					for j = 1; j <= n; j++ {
						for i = maxint(1, j-kd); i <= j; i++ {
							v.Set(kd+1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				} else {
					for j = 1; j <= n; j++ {
						for i = j; i <= minint(n, j+kd); i++ {
							v.Set(1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				}
				golapack.Zhbevx2stage('N', 'V', uplo, &n, &kd, v, ldu, u, ldu, &vl, &vu, &il, &iu, &abstol, &m3, wa3, z, ldu, work, lwork, rwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, KD=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHBEVX_2STAGE(N,V,"), uplo, ')'), iinfo, n, kd, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label930
					}
				}

				if m3 == 0 && n > 0 {
					result.Set(ntest-1, ulpinv)
					goto label930
				}

				//              Do test 36.
				temp1 = Dsxt1(func() *int { y := 1; return &y }(), wa2, &m2, wa3, &m3, &abstol, &ulp, &unfl)
				temp2 = Dsxt1(func() *int { y := 1; return &y }(), wa3, &m3, wa2, &m2, &abstol, &ulp, &unfl)
				if n > 0 {
					temp3 = maxf64(wa1.GetMag(0), wa1.GetMag(n-1))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/maxf64(unfl, temp3*ulp))

			label930:
				;

				//              Call ZHEEV
				golapack.Zlacpy(' ', &n, &n, a, lda, v, ldu)

				ntest = ntest + 1
				golapack.Zheev('V', uplo, &n, a, ldu, d1, work, lwork, rwork, &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHEEV(V,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label950
					}
				}

				//              Do tests 37 and 38
				Zhet21(func() *int { y := 1; return &y }(), uplo, &n, func() *int { y := 0; return &y }(), v, ldu, d1, d2, a, ldu, z, ldu, tau, work, rwork, result.Off(ntest-1))

				golapack.Zlacpy(' ', &n, &n, v, ldu, a, lda)

				ntest = ntest + 2
				golapack.Zheev2stage('N', uplo, &n, a, ldu, d3, work, lwork, rwork, &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHEEV_2STAGE(N,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label950
					}
				}

				//              Do test 39
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, d1.GetMag(j-1), d3.GetMag(j-1))
					temp2 = maxf64(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

			label950:
				;

				golapack.Zlacpy(' ', &n, &n, v, ldu, a, lda)

				//              Call ZHPEV
				//
				//              Load array WORK with the upper or lower triangular
				//              part of the matrix in packed form.
				if iuplo == 1 {
					indx = 1
					for j = 1; j <= n; j++ {
						for i = 1; i <= j; i++ {
							work.Set(indx-1, a.Get(i-1, j-1))
							indx = indx + 1
						}
					}
				} else {
					indx = 1
					for j = 1; j <= n; j++ {
						for i = j; i <= n; i++ {
							work.Set(indx-1, a.Get(i-1, j-1))
							indx = indx + 1
						}
					}
				}

				ntest = ntest + 1
				indwrk = n*(n+1)/2 + 1
				golapack.Zhpev('V', uplo, &n, work, d1, z, ldu, work.Off(indwrk-1), rwork, &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHPEV(V,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1050
					}
				}

				//              Do tests 40 and 41.
				Zhet21(func() *int { y := 1; return &y }(), uplo, &n, func() *int { y := 0; return &y }(), a, lda, d1, d2, z, ldu, v, ldu, tau, work, rwork, result.Off(ntest-1))

				if iuplo == 1 {
					indx = 1
					for j = 1; j <= n; j++ {
						for i = 1; i <= j; i++ {
							work.Set(indx-1, a.Get(i-1, j-1))
							indx = indx + 1
						}
					}
				} else {
					indx = 1
					for j = 1; j <= n; j++ {
						for i = j; i <= n; i++ {
							work.Set(indx-1, a.Get(i-1, j-1))
							indx = indx + 1
						}
					}
				}

				ntest = ntest + 2
				indwrk = n*(n+1)/2 + 1
				golapack.Zhpev('N', uplo, &n, work, d3, z, ldu, work.Off(indwrk-1), rwork, &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHPEV(N,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label1050
					}
				}

				//              Do test 42
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, d1.GetMag(j-1), d3.GetMag(j-1))
					temp2 = maxf64(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

			label1050:
				;

				//              Call ZHBEV
				if jtype <= 7 {
					kd = 0
				} else if jtype >= 8 && jtype <= 15 {
					kd = maxint(n-1, 0)
				} else {
					kd = ihbw
				}

				//              Load array V with the upper or lower triangular part
				//              of the matrix in band form.
				if iuplo == 1 {
					for j = 1; j <= n; j++ {
						for i = maxint(1, j-kd); i <= j; i++ {
							v.Set(kd+1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				} else {
					for j = 1; j <= n; j++ {
						for i = j; i <= minint(n, j+kd); i++ {
							v.Set(1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				}
				//
				ntest = ntest + 1
				golapack.Zhbev('V', uplo, &n, &kd, v, ldu, d1, z, ldu, work, rwork, &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, KD=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHBEV(V,"), uplo, ')'), iinfo, n, kd, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1140
					}
				}

				//              Do tests 43 and 44.
				Zhet21(func() *int { y := 1; return &y }(), uplo, &n, func() *int { y := 0; return &y }(), a, lda, d1, d2, z, ldu, v, ldu, tau, work, rwork, result.Off(ntest-1))

				if iuplo == 1 {
					for j = 1; j <= n; j++ {
						for i = maxint(1, j-kd); i <= j; i++ {
							v.Set(kd+1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				} else {
					for j = 1; j <= n; j++ {
						for i = j; i <= minint(n, j+kd); i++ {
							v.Set(1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				}

				ntest = ntest + 2
				golapack.Zhbev2stage('N', uplo, &n, &kd, v, ldu, d3, z, ldu, work, lwork, rwork, &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, KD=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHBEV_2STAGE(N,"), uplo, ')'), iinfo, n, kd, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label1140
					}
				}

			label1140:
				;

				//              Do test 45.
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, d1.GetMag(j-1), d3.GetMag(j-1))
					temp2 = maxf64(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

				golapack.Zlacpy(' ', &n, &n, a, lda, v, ldu)
				ntest = ntest + 1
				golapack.Zheevr('V', 'A', uplo, &n, a, ldu, &vl, &vu, &il, &iu, &abstol, &m, wa1, z, ldu, iwork, work, lwork, rwork, lrwork, toSlice(iwork, 2*n+1-1), toPtr((*liwork)-2*n), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHEEVR(V,A,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1170
					}
				}

				//              Do tests 45 and 46 (or ... )
				golapack.Zlacpy(' ', &n, &n, v, ldu, a, lda)

				Zhet21(func() *int { y := 1; return &y }(), uplo, &n, func() *int { y := 0; return &y }(), a, ldu, wa1, d2, z, ldu, v, ldu, tau, work, rwork, result.Off(ntest-1))

				ntest = ntest + 2
				golapack.Zheevr2stage('N', 'A', uplo, &n, a, ldu, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, iwork, work, lwork, rwork, lrwork, toSlice(iwork, 2*n+1-1), toPtr((*liwork)-2*n), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHEEVR_2STAGE(N,A,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label1170
					}
				}

				//              Do test 47 (or ... )
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, wa1.GetMag(j-1), wa2.GetMag(j-1))
					temp2 = maxf64(temp2, math.Abs(wa1.Get(j-1)-wa2.Get(j-1)))
				}
				result.Set(ntest-1, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

			label1170:
				;

				ntest = ntest + 1
				golapack.Zlacpy(' ', &n, &n, v, ldu, a, lda)
				golapack.Zheevr('V', 'I', uplo, &n, a, ldu, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, iwork, work, lwork, rwork, lrwork, toSlice(iwork, 2*n+1-1), toPtr((*liwork)-2*n), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHEEVR(V,I,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1180
					}
				}

				//              Do tests 48 and 49 (or +??)
				golapack.Zlacpy(' ', &n, &n, v, ldu, a, lda)

				Zhet22(func() *int { y := 1; return &y }(), uplo, &n, &m2, func() *int { y := 0; return &y }(), a, ldu, wa2, d2, z, ldu, v, ldu, tau, work, rwork, result.Off(ntest-1))

				ntest = ntest + 2
				golapack.Zlacpy(' ', &n, &n, v, ldu, a, lda)
				golapack.Zheevr2stage('N', 'I', uplo, &n, a, ldu, &vl, &vu, &il, &iu, &abstol, &m3, wa3, z, ldu, iwork, work, lwork, rwork, lrwork, toSlice(iwork, 2*n+1-1), toPtr((*liwork)-2*n), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHEEVR_2STAGE(N,I,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label1180
					}
				}

				//              Do test 50 (or +??)
				temp1 = Dsxt1(func() *int { y := 1; return &y }(), wa2, &m2, wa3, &m3, &abstol, &ulp, &unfl)
				temp2 = Dsxt1(func() *int { y := 1; return &y }(), wa3, &m3, wa2, &m2, &abstol, &ulp, &unfl)
				result.Set(ntest-1, (temp1+temp2)/maxf64(unfl, ulp*temp3))
			label1180:
				;

				ntest = ntest + 1
				golapack.Zlacpy(' ', &n, &n, v, ldu, a, lda)
				golapack.Zheevr('V', 'V', uplo, &n, a, ldu, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, iwork, work, lwork, rwork, lrwork, toSlice(iwork, 2*n+1-1), toPtr((*liwork)-2*n), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHEEVR(V,V,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1190
					}
				}

				//              Do tests 51 and 52 (or +??)
				golapack.Zlacpy(' ', &n, &n, v, ldu, a, lda)

				Zhet22(func() *int { y := 1; return &y }(), uplo, &n, &m2, func() *int { y := 0; return &y }(), a, ldu, wa2, d2, z, ldu, v, ldu, tau, work, rwork, result.Off(ntest-1))

				ntest = ntest + 2
				golapack.Zlacpy(' ', &n, &n, v, ldu, a, lda)
				golapack.Zheevr2stage('N', 'V', uplo, &n, a, ldu, &vl, &vu, &il, &iu, &abstol, &m3, wa3, z, ldu, iwork, work, lwork, rwork, lrwork, toSlice(iwork, 2*n+1-1), toPtr((*liwork)-2*n), &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVST2STG: %s returned INFO=%6d\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", append([]byte("ZHEEVR_2STAGE(N,V,"), uplo, ')'), iinfo, n, jtype, ioldsd)
					(*info) = absint(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label1190
					}
				}

				if m3 == 0 && n > 0 {
					result.Set(ntest-1, ulpinv)
					goto label1190
				}

				//              Do test 52 (or +??)
				temp1 = Dsxt1(func() *int { y := 1; return &y }(), wa2, &m2, wa3, &m3, &abstol, &ulp, &unfl)
				temp2 = Dsxt1(func() *int { y := 1; return &y }(), wa3, &m3, wa2, &m2, &abstol, &ulp, &unfl)
				if n > 0 {
					temp3 = maxf64(wa1.GetMag(0), wa1.GetMag(n-1))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/maxf64(unfl, temp3*ulp))

				golapack.Zlacpy(' ', &n, &n, v, ldu, a, lda)

				//              Load array V with the upper or lower triangular part
				//              of the matrix in band form.
			label1190:
			}

			//           End of Loop -- Check for RESULT(j) > THRESH
			ntestt = ntestt + ntest
			Dlafts([]byte("ZST"), &n, &n, &jtype, &ntest, result, &ioldsd, thresh, &nerrs, t)

		label1210:
		}
	}

	//     Summary
	Alasvm([]byte("ZST"), &nerrs, &ntestt, func() *int { y := 0; return &y }())
}
