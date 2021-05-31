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

// Ddrvst2stg checks the symmetric eigenvalue problem drivers.
//
//              DSTEV computes all eigenvalues and, optionally,
//              eigenvectors of a real symmetric tridiagonal matrix.
//
//              DSTEVX computes selected eigenvalues and, optionally,
//              eigenvectors of a real symmetric tridiagonal matrix.
//
//              DSTEVR computes selected eigenvalues and, optionally,
//              eigenvectors of a real symmetric tridiagonal matrix
//              using the Relatively Robust Representation where it can.
//
//              DSYEV computes all eigenvalues and, optionally,
//              eigenvectors of a real symmetric matrix.
//
//              DSYEVX computes selected eigenvalues and, optionally,
//              eigenvectors of a real symmetric matrix.
//
//              DSYEVR computes selected eigenvalues and, optionally,
//              eigenvectors of a real symmetric matrix
//              using the Relatively Robust Representation where it can.
//
//              DSPEV computes all eigenvalues and, optionally,
//              eigenvectors of a real symmetric matrix in packed
//              storage.
//
//              DSPEVX computes selected eigenvalues and, optionally,
//              eigenvectors of a real symmetric matrix in packed
//              storage.
//
//              DSBEV computes all eigenvalues and, optionally,
//              eigenvectors of a real symmetric band matrix.
//
//              DSBEVX computes selected eigenvalues and, optionally,
//              eigenvectors of a real symmetric band matrix.
//
//              DSYEVD computes all eigenvalues and, optionally,
//              eigenvectors of a real symmetric matrix using
//              a divide and conquer algorithm.
//
//              DSPEVD computes all eigenvalues and, optionally,
//              eigenvectors of a real symmetric matrix in packed
//              storage, using a divide and conquer algorithm.
//
//              DSBEVD computes all eigenvalues and, optionally,
//              eigenvectors of a real symmetric band matrix,
//              using a divide and conquer algorithm.
//
//      When DDRVST2STG is called, a number of matrix "sizes" ("n's") and a
//      number of matrix "types" are specified.  For each size ("n")
//      and each type of matrix, one matrix will be generated and used
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
//      if DOTYPE(j) is .TRUE., then matrix type "j" will be generated.
//      Currently, the list of possible types is:
//
//      (1)  The zero matrix.
//      (2)  The identity matrix.
//
//      (3)  A diagonal matrix with evenly spaced eigenvalues
//           1, ..., ULP  and random signs.
//           (ULP = (first number larger than 1) - 1 )
//      (4)  A diagonal matrix with geometrically spaced eigenvalues
//           1, ..., ULP  and random signs.
//      (5)  A diagonal matrix with "clustered" eigenvalues
//           1, ULP, ..., ULP and random signs.
//
//      (6)  Same as (4), but multiplied by SQRT( overflow threshold )
//      (7)  Same as (4), but multiplied by SQRT( underflow threshold )
//
//      (8)  A matrix of the form  U' D U, where U is orthogonal and
//           D has evenly spaced entries 1, ..., ULP with random signs
//           on the diagonal.
//
//      (9)  A matrix of the form  U' D U, where U is orthogonal and
//           D has geometrically spaced entries 1, ..., ULP with random
//           signs on the diagonal.
//
//      (10) A matrix of the form  U' D U, where U is orthogonal and
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
func Ddrvst2stg(nsizes *int, nn *[]int, ntypes *int, dotype *[]bool, iseed *[]int, thresh *float64, nounit *int, a *mat.Matrix, lda *int, d1, d2, d3, d4, eveigs, wa1, wa2, wa3 *mat.Vector, u *mat.Matrix, ldu *int, v *mat.Matrix, tau *mat.Vector, z *mat.Matrix, work *mat.Vector, lwork *int, iwork *[]int, liwork *int, result *mat.Vector, info *int, t *testing.T) {
	var badnn bool
	var uplo byte
	var abstol, aninv, anorm, cond, half, one, ovfl, rtovfl, rtunfl, temp1, temp2, temp3, ten, two, ulp, ulpinv, unfl, vl, vu, zero float64
	var i, idiag, ihbw, iinfo, il, imode, indx, irow, itemp, itype, iu, iuplo, j, j1, j2, jcol, jsize, jtype, kd, lgn, liwedc, lwedc, m, m2, m3, maxtyp, mtypes, n, nerrs, nmats, nmax, ntest, ntestt int
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
	half = 0.5
	maxtyp = 18
	srnamt := &gltest.Common.Srnamc.Srnamt

	ktype[0], ktype[1], ktype[2], ktype[3], ktype[4], ktype[5], ktype[6], ktype[7], ktype[8], ktype[9], ktype[10], ktype[11], ktype[12], ktype[13], ktype[14], ktype[15], ktype[16], ktype[17] = 1, 2, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 8, 8, 8, 9, 9, 9
	kmagn[0], kmagn[1], kmagn[2], kmagn[3], kmagn[4], kmagn[5], kmagn[6], kmagn[7], kmagn[8], kmagn[9], kmagn[10], kmagn[11], kmagn[12], kmagn[13], kmagn[14], kmagn[15], kmagn[16], kmagn[17] = 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3
	kmode[0], kmode[1], kmode[2], kmode[3], kmode[4], kmode[5], kmode[6], kmode[7], kmode[8], kmode[9], kmode[10], kmode[11], kmode[12], kmode[13], kmode[14], kmode[15], kmode[16], kmode[17] = 0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0, 4, 4, 4

	//     Keep ftrnchek happy
	vl = zero
	vu = zero

	//     1)      Check for errors
	ntestt = 0
	*info = 0

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
		*info = -1
	} else if badnn {
		*info = -2
	} else if (*ntypes) < 0 {
		*info = -3
	} else if (*lda) < nmax {
		*info = -9
	} else if (*ldu) < nmax {
		*info = -16
	} else if 2*int(math.Pow(float64(maxint(2, nmax)), 2)) > (*lwork) {
		*info = -21
	}

	if *info != 0 {
		gltest.Xerbla([]byte("DDRVST2STG"), -*info)
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
			if int(math.Pow(2, float64(lgn))) < n {
				lgn = lgn + 1
			}
			if int(math.Pow(2, float64(lgn))) < n {
				lgn = lgn + 1
			}
			lwedc = 1 + 4*n + 2*n*lgn + 4*int(math.Pow(float64(n), 2))
			//c           LIWEDC = 6 + 6*N + 5*N*LGN
			liwedc = 3 + 5*n
		} else {
			lwedc = 9
			//c           LIWEDC = 12
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
				goto label1730
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
			//           =5         random log   symmetric, w/ eigenvalues
			//           =6         random       (none)
			//           =7                      random diagonal
			//           =8                      random symmetric
			//           =9                      band symmetric, w/ eigenvalues
			//
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

			golapack.Dlaset('F', lda, &n, &zero, &zero, a, lda)
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
					a.Set(jcol-1, jcol-1, anorm)
				}

			} else if itype == 4 {
				//              Diagonal Matrix, [Eigen]values Specified
				matgen.Dlatms(&n, &n, 'S', iseed, 'S', work, &imode, &cond, &anorm, toPtr(0), toPtr(0), 'N', a, lda, work.Off(n+1-1), &iinfo)

			} else if itype == 5 {
				//              Symmetric, eigenvalues specified
				matgen.Dlatms(&n, &n, 'S', iseed, 'S', work, &imode, &cond, &anorm, &n, &n, 'N', a, lda, work.Off(n+1-1), &iinfo)

			} else if itype == 7 {
				//              Diagonal, random eigenvalues
				idumma[0] = 1
				matgen.Dlatmr(&n, &n, 'S', iseed, 'S', work, toPtr(6), &one, &one, 'T', 'N', work.Off(n+1-1), toPtr(1), &one, work.Off(2*n+1-1), toPtr(1), &one, 'N', &idumma, toPtr(0), toPtr(0), &zero, &anorm, 'N', a, lda, iwork, &iinfo)

			} else if itype == 8 {
				//              Symmetric, random eigenvalues
				idumma[0] = 1
				matgen.Dlatmr(&n, &n, 'S', iseed, 'S', work, toPtr(6), &one, &one, 'T', 'N', work.Off(n+1-1), toPtr(1), &one, work.Off(2*n+1-1), toPtr(1), &one, 'N', &idumma, &n, &n, &zero, &anorm, 'N', a, lda, iwork, &iinfo)

			} else if itype == 9 {
				//              Symmetric banded, eigenvalues specified
				ihbw = int(float64(n-1) * matgen.Dlarnd(toPtr(1), &iseed3))
				matgen.Dlatms(&n, &n, 'S', iseed, 'S', work, &imode, &cond, &anorm, &ihbw, &ihbw, 'Z', u, ldu, work.Off(n+1-1), &iinfo)

				//              Store as dense matrix for most routines.
				golapack.Dlaset('F', lda, &n, &zero, &zero, a, lda)
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
				fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "Generator", iinfo, n, jtype, ioldsd)
				*info = int(math.Abs(float64(iinfo)))
				return
			}

		label110:
			;

			abstol = unfl + unfl
			if n <= 1 {
				il = 1
				iu = n
			} else {
				il = 1 + int(float64(n-1)*matgen.Dlarnd(toPtr(1), &iseed2))
				iu = 1 + int(float64(n-1)*matgen.Dlarnd(toPtr(1), &iseed2))
				if il > iu {
					itemp = il
					il = iu
					iu = itemp
				}
			}

			//           3)      If matrix is tridiagonal, call DSTEV and DSTEVX.
			if jtype <= 7 {
				ntest = 1
				for i = 1; i <= n; i++ {
					d1.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d2.Set(i-1, float64(a.Get(i+1-1, i-1)))
				}
				*srnamt = "DSTEV"
				golapack.Dstev('V', &n, d1, d2, z, ldu, work, &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEV(V)", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(0, ulpinv)
						result.Set(1, ulpinv)
						result.Set(2, ulpinv)
						goto label180
					}
				}

				//              Do tests 1 and 2.
				for i = 1; i <= n; i++ {
					d3.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i+1-1, i-1)))
				}
				Dstt21(&n, toPtr(0), d3, d4, d1, d2, z, ldu, work, result.Off(0))

				ntest = 3
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i+1-1, i-1)))
				}
				*srnamt = "DSTEV"
				golapack.Dstev('N', &n, d3, d4, z, ldu, work, &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEV(N)", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(2, ulpinv)
						goto label180
					}
				}

				//              Do test 3.
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, math.Abs(d1.Get(j-1)), math.Abs(d3.Get(j-1)))
					temp2 = maxf64(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(2, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

			label180:
				;

				ntest = 4
				for i = 1; i <= n; i++ {
					eveigs.Set(i-1, d3.Get(i-1))
					d1.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d2.Set(i-1, float64(a.Get(i+1-1, i-1)))
				}
				*srnamt = "DSTEVX"
				golapack.Dstevx('V', 'A', &n, d1, d2, &vl, &vu, &il, &iu, &abstol, &m, wa1, z, ldu, work, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEVX(V,A)", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(3, ulpinv)
						result.Set(4, ulpinv)
						result.Set(5, ulpinv)
						goto label250
					}
				}
				if n > 0 {
					temp3 = maxf64(math.Abs(wa1.Get(0)), math.Abs(wa1.Get(n-1)))
				} else {
					temp3 = zero
				}

				//              Do tests 4 and 5.
				for i = 1; i <= n; i++ {
					d3.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i+1-1, i-1)))
				}
				Dstt21(&n, toPtr(0), d3, d4, wa1, d2, z, ldu, work, result.Off(3))

				ntest = 6
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i+1-1, i-1)))
				}
				*srnamt = "DSTEVX"
				golapack.Dstevx('N', 'A', &n, d3, d4, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, work, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEVX(N,A)", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(5, ulpinv)
						goto label250
					}
				}

				//              Do test 6.
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, math.Abs(wa2.Get(j-1)), math.Abs(eveigs.Get(j-1)))
					temp2 = maxf64(temp2, math.Abs(wa2.Get(j-1)-eveigs.Get(j-1)))
				}
				result.Set(5, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

			label250:
				;

				ntest = 7
				for i = 1; i <= n; i++ {
					d1.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d2.Set(i-1, float64(a.Get(i+1-1, i-1)))
				}
				*srnamt = "DSTEVR"
				golapack.Dstevr('V', 'A', &n, d1, d2, &vl, &vu, &il, &iu, &abstol, &m, wa1, z, ldu, iwork, work, lwork, toSlice(iwork, 2*n+1-1), toPtr((*liwork)-2*n), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEVR(V,A)", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(6, ulpinv)
						result.Set(7, ulpinv)
						goto label320
					}
				}
				if n > 0 {
					temp3 = maxf64(math.Abs(wa1.Get(0)), math.Abs(wa1.Get(n-1)))
				} else {
					temp3 = zero
				}

				//              Do tests 7 and 8.
				for i = 1; i <= n; i++ {
					d3.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i+1-1, i-1)))
				}
				Dstt21(&n, toPtr(0), d3, d4, wa1, d2, z, ldu, work, result.Off(6))

				ntest = 9
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i+1-1, i-1)))
				}
				*srnamt = "DSTEVR"
				golapack.Dstevr('N', 'A', &n, d3, d4, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, iwork, work, lwork, toSlice(iwork, 2*n+1-1), toPtr((*liwork)-2*n), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEVR(N,A)", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(8, ulpinv)
						goto label320
					}
				}

				//              Do test 9.
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, math.Abs(wa2.Get(j-1)), math.Abs(eveigs.Get(j-1)))
					temp2 = maxf64(temp2, math.Abs(wa2.Get(j-1)-eveigs.Get(j-1)))
				}
				result.Set(8, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

			label320:
				;

				ntest = 10
				for i = 1; i <= n; i++ {
					d1.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d2.Set(i-1, float64(a.Get(i+1-1, i-1)))
				}
				*srnamt = "DSTEVX"
				golapack.Dstevx('V', 'I', &n, d1, d2, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, work, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEVX(V,I)", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(9, ulpinv)
						result.Set(10, ulpinv)
						result.Set(11, ulpinv)
						goto label380
					}
				}

				//              Do tests 10 and 11.
				for i = 1; i <= n; i++ {
					d3.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i+1-1, i-1)))
				}
				Dstt22(&n, &m2, toPtr(0), d3, d4, wa2, d2, z, ldu, work.Matrix(maxint(1, m2), opts), toPtr(maxint(1, m2)), result.Off(9))

				ntest = 12
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i+1-1, i-1)))
				}
				*srnamt = "DSTEVX"
				golapack.Dstevx('N', 'I', &n, d3, d4, &vl, &vu, &il, &iu, &abstol, &m3, wa3, z, ldu, work, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEVX(N,I)", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(11, ulpinv)
						goto label380
					}
				}

				//              Do test 12.
				temp1 = Dsxt1(toPtr(1), wa2, &m2, wa3, &m3, &abstol, &ulp, &unfl)
				temp2 = Dsxt1(toPtr(1), wa3, &m3, wa2, &m2, &abstol, &ulp, &unfl)
				result.Set(11, (temp1+temp2)/maxf64(unfl, ulp*temp3))

			label380:
				;

				ntest = 12
				if n > 0 {
					if il != 1 {
						vl = wa1.Get(il-1) - maxf64(half*(wa1.Get(il-1)-wa1.Get(il-1-1)), ten*ulp*temp3, ten*rtunfl)
					} else {
						vl = wa1.Get(0) - maxf64(half*(wa1.Get(n-1)-wa1.Get(0)), ten*ulp*temp3, ten*rtunfl)
					}
					if iu != n {
						vu = wa1.Get(iu-1) + maxf64(half*(wa1.Get(iu+1-1)-wa1.Get(iu-1)), ten*ulp*temp3, ten*rtunfl)
					} else {
						vu = wa1.Get(n-1) + maxf64(half*(wa1.Get(n-1)-wa1.Get(0)), ten*ulp*temp3, ten*rtunfl)
					}
				} else {
					vl = zero
					vu = one
				}

				for i = 1; i <= n; i++ {
					d1.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d2.Set(i-1, float64(a.Get(i+1-1, i-1)))
				}
				*srnamt = "DSTEVX"
				golapack.Dstevx('V', 'V', &n, d1, d2, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, work, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEVX(V,V)", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(12, ulpinv)
						result.Set(13, ulpinv)
						result.Set(14, ulpinv)
						goto label440
					}
				}

				if m2 == 0 && n > 0 {
					result.Set(12, ulpinv)
					result.Set(13, ulpinv)
					result.Set(14, ulpinv)
					goto label440
				}

				//              Do tests 13 and 14.
				for i = 1; i <= n; i++ {
					d3.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i+1-1, i-1)))
				}
				Dstt22(&n, &m2, toPtr(0), d3, d4, wa2, d2, z, ldu, work.Matrix(maxint(1, m2), opts), toPtr(maxint(1, m2)), result.Off(12))

				ntest = 15
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i+1-1, i-1)))
				}
				*srnamt = "DSTEVX"
				golapack.Dstevx('N', 'V', &n, d3, d4, &vl, &vu, &il, &iu, &abstol, &m3, wa3, z, ldu, work, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEVX(N,V)", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(14, ulpinv)
						goto label440
					}
				}

				//              Do test 15.
				temp1 = Dsxt1(toPtr(1), wa2, &m2, wa3, &m3, &abstol, &ulp, &unfl)
				temp2 = Dsxt1(toPtr(1), wa3, &m3, wa2, &m2, &abstol, &ulp, &unfl)
				result.Set(14, (temp1+temp2)/maxf64(unfl, temp3*ulp))

			label440:
				;

				ntest = 16
				for i = 1; i <= n; i++ {
					d1.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d2.Set(i-1, float64(a.Get(i+1-1, i-1)))
				}
				*srnamt = "DSTEVD"
				golapack.Dstevd('V', &n, d1, d2, z, ldu, work, &lwedc, iwork, &liwedc, &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEVD(V)", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(15, ulpinv)
						result.Set(16, ulpinv)
						result.Set(17, ulpinv)
						goto label510
					}
				}

				//              Do tests 16 and 17.
				for i = 1; i <= n; i++ {
					d3.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i+1-1, i-1)))
				}
				Dstt21(&n, toPtr(0), d3, d4, d1, d2, z, ldu, work, result.Off(15))

				ntest = 18
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i+1-1, i-1)))
				}
				*srnamt = "DSTEVD"
				golapack.Dstevd('N', &n, d3, d4, z, ldu, work, &lwedc, iwork, &liwedc, &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEVD(N)", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(17, ulpinv)
						goto label510
					}
				}

				//              Do test 18.
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, math.Abs(eveigs.Get(j-1)), math.Abs(d3.Get(j-1)))
					temp2 = maxf64(temp2, math.Abs(eveigs.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(17, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

			label510:
				;

				ntest = 19
				for i = 1; i <= n; i++ {
					d1.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d2.Set(i-1, float64(a.Get(i+1-1, i-1)))
				}
				*srnamt = "DSTEVR"
				golapack.Dstevr('V', 'I', &n, d1, d2, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, iwork, work, lwork, toSlice(iwork, 2*n+1-1), toPtr((*liwork)-2*n), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEVR(V,I)", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(18, ulpinv)
						result.Set(19, ulpinv)
						result.Set(20, ulpinv)
						goto label570
					}
				}

				//              DO tests 19 and 20.
				for i = 1; i <= n; i++ {
					d3.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i+1-1, i-1)))
				}
				Dstt22(&n, &m2, toPtr(0), d3, d4, wa2, d2, z, ldu, work.Matrix(maxint(1, m2), opts), toPtr(maxint(1, m2)), result.Off(18))

				ntest = 21
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i+1-1, i-1)))
				}
				*srnamt = "DSTEVR"
				golapack.Dstevr('N', 'I', &n, d3, d4, &vl, &vu, &il, &iu, &abstol, &m3, wa3, z, ldu, iwork, work, lwork, toSlice(iwork, 2*n+1-1), toPtr((*liwork)-2*n), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEVR(N,I)", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(20, ulpinv)
						goto label570
					}
				}

				//              Do test 21.
				temp1 = Dsxt1(toPtr(1), wa2, &m2, wa3, &m3, &abstol, &ulp, &unfl)
				temp2 = Dsxt1(toPtr(1), wa3, &m3, wa2, &m2, &abstol, &ulp, &unfl)
				result.Set(20, (temp1+temp2)/maxf64(unfl, ulp*temp3))

			label570:
				;

				ntest = 21
				if n > 0 {
					if il != 1 {
						vl = wa1.Get(il-1) - maxf64(half*(wa1.Get(il-1)-wa1.Get(il-1-1)), ten*ulp*temp3, ten*rtunfl)
					} else {
						vl = wa1.Get(0) - maxf64(half*(wa1.Get(n-1)-wa1.Get(0)), ten*ulp*temp3, ten*rtunfl)
					}
					if iu != n {
						vu = wa1.Get(iu-1) + maxf64(half*(wa1.Get(iu+1-1)-wa1.Get(iu-1)), ten*ulp*temp3, ten*rtunfl)
					} else {
						vu = wa1.Get(n-1) + maxf64(half*(wa1.Get(n-1)-wa1.Get(0)), ten*ulp*temp3, ten*rtunfl)
					}
				} else {
					vl = zero
					vu = one
				}

				for i = 1; i <= n; i++ {
					d1.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d2.Set(i-1, float64(a.Get(i+1-1, i-1)))
				}
				*srnamt = "DSTEVR"
				golapack.Dstevr('V', 'V', &n, d1, d2, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, iwork, work, lwork, toSlice(iwork, 2*n+1-1), toPtr((*liwork)-2*n), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEVR(V,V)", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(21, ulpinv)
						result.Set(22, ulpinv)
						result.Set(23, ulpinv)
						goto label630
					}
				}

				if m2 == 0 && n > 0 {
					result.Set(21, ulpinv)
					result.Set(22, ulpinv)
					result.Set(23, ulpinv)
					goto label630
				}

				//              Do tests 22 and 23.
				for i = 1; i <= n; i++ {
					d3.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i+1-1, i-1)))
				}
				Dstt22(&n, &m2, toPtr(0), d3, d4, wa2, d2, z, ldu, work.Matrix(maxint(1, m2), opts), toPtr(maxint(1, m2)), result.Off(21))

				ntest = 24
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i+1-1, i-1)))
				}
				*srnamt = "DSTEVR"
				golapack.Dstevr('N', 'V', &n, d3, d4, &vl, &vu, &il, &iu, &abstol, &m3, wa3, z, ldu, iwork, work, lwork, toSlice(iwork, 2*n+1-1), toPtr((*liwork)-2*n), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSTEVR(N,V)", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(23, ulpinv)
						goto label630
					}
				}

				//              Do test 24.
				temp1 = Dsxt1(toPtr(1), wa2, &m2, wa3, &m3, &abstol, &ulp, &unfl)
				temp2 = Dsxt1(toPtr(1), wa3, &m3, wa2, &m2, &abstol, &ulp, &unfl)
				result.Set(23, (temp1+temp2)/maxf64(unfl, temp3*ulp))

			label630:
			} else {
				for i = 1; i <= 24; i++ {
					result.Set(i-1, zero)
				}
				ntest = 24
			}

			//           Perform remaining tests storing upper or lower triangular
			//           part of matrix.
			for iuplo = 0; iuplo <= 1; iuplo++ {
				if iuplo == 0 {
					uplo = 'L'
				} else {
					uplo = 'U'
				}

				//              4)      Call DSYEV and DSYEVX.
				golapack.Dlacpy(' ', &n, &n, a, lda, v, ldu)

				ntest = ntest + 1
				*srnamt = "DSYEV"
				golapack.Dsyev('V', uplo, &n, a, ldu, d1, work, lwork, &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSYEV(V,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label660
					}
				}

				//              Do tests 25 and 26 (or +54)
				Dsyt21(toPtr(1), uplo, &n, toPtr(0), v, ldu, d1, d2, a, ldu, z, ldu, tau, work, result.Off(ntest-1))

				golapack.Dlacpy(' ', &n, &n, v, ldu, a, lda)

				ntest = ntest + 2
				*srnamt = "DSYEV_2STAGE"
				golapack.Dsyev2stage('N', uplo, &n, a, ldu, d3, work, lwork, &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSYEV_2STAGE(N,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label660
					}
				}

				//              Do test 27 (or +54)
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, math.Abs(d1.Get(j-1)), math.Abs(d3.Get(j-1)))
					temp2 = maxf64(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

			label660:
				;
				golapack.Dlacpy(' ', &n, &n, v, ldu, a, lda)

				ntest = ntest + 1

				if n > 0 {
					temp3 = maxf64(math.Abs(d1.Get(0)), math.Abs(d1.Get(n-1)))
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

				*srnamt = "DSYEVX"
				golapack.Dsyevx('V', 'A', uplo, &n, a, ldu, &vl, &vu, &il, &iu, &abstol, &m, wa1, z, ldu, work, lwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSYEVX(V,A,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label680
					}
				}

				//              Do tests 28 and 29 (or +54)
				golapack.Dlacpy(' ', &n, &n, v, ldu, a, lda)

				Dsyt21(toPtr(1), uplo, &n, toPtr(0), a, ldu, d1, d2, z, ldu, v, ldu, tau, work, result.Off(ntest-1))

				ntest = ntest + 2
				*srnamt = "DSYEVX_2STAGE"
				golapack.Dsyevx2stage('N', 'A', uplo, &n, a, ldu, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, work, lwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSYEVX_2STAGE(N,A,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label680
					}
				}

				//              Do test 30 (or +54)
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, math.Abs(wa1.Get(j-1)), math.Abs(wa2.Get(j-1)))
					temp2 = maxf64(temp2, math.Abs(wa1.Get(j-1)-wa2.Get(j-1)))
				}
				result.Set(ntest-1, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

			label680:
				;

				ntest = ntest + 1
				golapack.Dlacpy(' ', &n, &n, v, ldu, a, lda)
				*srnamt = "DSYEVX"
				golapack.Dsyevx('V', 'I', uplo, &n, a, ldu, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, work, lwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSYEVX(V,I,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label690
					}
				}

				//              Do tests 31 and 32 (or +54)
				golapack.Dlacpy(' ', &n, &n, v, ldu, a, lda)

				Dsyt22(toPtr(1), uplo, &n, &m2, toPtr(0), a, ldu, wa2, d2, z, ldu, v, ldu, tau, work, result.Off(ntest-1))

				ntest = ntest + 2
				golapack.Dlacpy(' ', &n, &n, v, ldu, a, lda)
				*srnamt = "DSYEVX_2STAGE"
				golapack.Dsyevx2stage('N', 'I', uplo, &n, a, ldu, &vl, &vu, &il, &iu, &abstol, &m3, wa3, z, ldu, work, lwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSYEVX_2STAGE(N,I,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label690
					}
				}

				//              Do test 33 (or +54)
				temp1 = Dsxt1(toPtr(1), wa2, &m2, wa3, &m3, &abstol, &ulp, &unfl)
				temp2 = Dsxt1(toPtr(1), wa3, &m3, wa2, &m2, &abstol, &ulp, &unfl)
				result.Set(ntest-1, (temp1+temp2)/maxf64(unfl, ulp*temp3))
			label690:
				;

				ntest = ntest + 1
				golapack.Dlacpy(' ', &n, &n, v, ldu, a, lda)
				*srnamt = "DSYEVX"
				golapack.Dsyevx('V', 'V', uplo, &n, a, ldu, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, work, lwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSYEVX(V,V,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label700
					}
				}

				//              Do tests 34 and 35 (or +54)
				golapack.Dlacpy(' ', &n, &n, v, ldu, a, lda)

				Dsyt22(toPtr(1), uplo, &n, &m2, toPtr(0), a, ldu, wa2, d2, z, ldu, v, ldu, tau, work, result.Off(ntest-1))

				ntest = ntest + 2
				golapack.Dlacpy(' ', &n, &n, v, ldu, a, lda)
				*srnamt = "DSYEVX_2STAGE"
				golapack.Dsyevx2stage('N', 'V', uplo, &n, a, ldu, &vl, &vu, &il, &iu, &abstol, &m3, wa3, z, ldu, work, lwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSYEVX_2STAGE(N,V,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label700
					}
				}

				if m3 == 0 && n > 0 {
					result.Set(ntest-1, ulpinv)
					goto label700
				}

				//              Do test 36 (or +54)
				temp1 = Dsxt1(toPtr(1), wa2, &m2, wa3, &m3, &abstol, &ulp, &unfl)
				temp2 = Dsxt1(toPtr(1), wa3, &m3, wa2, &m2, &abstol, &ulp, &unfl)
				if n > 0 {
					temp3 = maxf64(math.Abs(wa1.Get(0)), math.Abs(wa1.Get(n-1)))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/maxf64(unfl, temp3*ulp))

			label700:
				;

				//              5)      Call DSPEV and DSPEVX.
				golapack.Dlacpy(' ', &n, &n, v, ldu, a, lda)

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
				*srnamt = "DSPEV"
				golapack.Dspev('V', uplo, &n, work, d1, z, ldu, v.VectorIdx(0), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSPEV(V,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label800
					}
				}

				//              Do tests 37 and 38 (or +54)
				Dsyt21(toPtr(1), uplo, &n, toPtr(0), a, lda, d1, d2, z, ldu, v, ldu, tau, work, result.Off(ntest-1))

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
				*srnamt = "DSPEV"
				golapack.Dspev('N', uplo, &n, work, d3, z, ldu, v.VectorIdx(0), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSPEV(N,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label800
					}
				}

				//              Do test 39 (or +54)
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, math.Abs(d1.Get(j-1)), math.Abs(d3.Get(j-1)))
					temp2 = maxf64(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

				//              Load array WORK with the upper or lower triangular part
				//              of the matrix in packed form.
			label800:
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
					temp3 = maxf64(math.Abs(d1.Get(0)), math.Abs(d1.Get(n-1)))
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

				*srnamt = "DSPEVX"
				golapack.Dspevx('V', 'A', uplo, &n, work, &vl, &vu, &il, &iu, &abstol, &m, wa1, z, ldu, v.VectorIdx(0), iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSPEVX(V,A,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label900
					}
				}

				//              Do tests 40 and 41 (or +54)
				Dsyt21(toPtr(1), uplo, &n, toPtr(0), a, ldu, wa1, d2, z, ldu, v, ldu, tau, work, result.Off(ntest-1))

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

				*srnamt = "DSPEVX"
				golapack.Dspevx('N', 'A', uplo, &n, work, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, v.VectorIdx(0), iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSPEVX(N,A,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label900
					}
				}

				//              Do test 42 (or +54)
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, math.Abs(wa1.Get(j-1)), math.Abs(wa2.Get(j-1)))
					temp2 = maxf64(temp2, math.Abs(wa1.Get(j-1)-wa2.Get(j-1)))
				}
				result.Set(ntest-1, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

			label900:
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

				*srnamt = "DSPEVX"
				golapack.Dspevx('V', 'I', uplo, &n, work, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, v.VectorIdx(0), iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSPEVX(V,I,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label990
					}
				}

				//              Do tests 43 and 44 (or +54)
				Dsyt22(toPtr(1), uplo, &n, &m2, toPtr(0), a, ldu, wa2, d2, z, ldu, v, ldu, tau, work, result.Off(ntest-1))

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

				*srnamt = "DSPEVX"
				golapack.Dspevx('N', 'I', uplo, &n, work, &vl, &vu, &il, &iu, &abstol, &m3, wa3, z, ldu, v.VectorIdx(0), iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSPEVX(N,I,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label990
					}
				}

				if m3 == 0 && n > 0 {
					result.Set(ntest-1, ulpinv)
					goto label990
				}

				//              Do test 45 (or +54)
				temp1 = Dsxt1(toPtr(1), wa2, &m2, wa3, &m3, &abstol, &ulp, &unfl)
				temp2 = Dsxt1(toPtr(1), wa3, &m3, wa2, &m2, &abstol, &ulp, &unfl)
				if n > 0 {
					temp3 = maxf64(math.Abs(wa1.Get(0)), math.Abs(wa1.Get(n-1)))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/maxf64(unfl, temp3*ulp))

			label990:
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

				*srnamt = "DSPEVX"
				golapack.Dspevx('V', 'V', uplo, &n, work, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, v.VectorIdx(0), iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSPEVX(V,V,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1080
					}
				}

				//              Do tests 46 and 47 (or +54)
				Dsyt22(toPtr(1), uplo, &n, &m2, toPtr(0), a, ldu, wa2, d2, z, ldu, v, ldu, tau, work, result.Off(ntest-1))

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

				*srnamt = "DSPEVX"
				golapack.Dspevx('N', 'V', uplo, &n, work, &vl, &vu, &il, &iu, &abstol, &m3, wa3, z, ldu, v.VectorIdx(0), iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSPEVX(N,V,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label1080
					}
				}

				if m3 == 0 && n > 0 {
					result.Set(ntest-1, ulpinv)
					goto label1080
				}

				//              Do test 48 (or +54)
				temp1 = Dsxt1(toPtr(1), wa2, &m2, wa3, &m3, &abstol, &ulp, &unfl)
				temp2 = Dsxt1(toPtr(1), wa3, &m3, wa2, &m2, &abstol, &ulp, &unfl)
				if n > 0 {
					temp3 = maxf64(math.Abs(wa1.Get(0)), math.Abs(wa1.Get(n-1)))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/maxf64(unfl, temp3*ulp))

			label1080:
				;

				//              6)      Call DSBEV and DSBEVX.
				if jtype <= 7 {
					kd = 1
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
				*srnamt = "DSBEV"
				golapack.Dsbev('V', uplo, &n, &kd, v, ldu, d1, z, ldu, work, &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSBEV(V,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1180
					}
				}

				//              Do tests 49 and 50 (or ... )
				Dsyt21(toPtr(1), uplo, &n, toPtr(0), a, lda, d1, d2, z, ldu, v, ldu, tau, work, result.Off(ntest-1))

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
				*srnamt = "DSBEV_2STAGE"
				golapack.Dsbev2stage('N', uplo, &n, &kd, v, ldu, d3, z, ldu, work, lwork, &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSBEV_2STAGE(N,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label1180
					}
				}

				//              Do test 51 (or +54)
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, math.Abs(d1.Get(j-1)), math.Abs(d3.Get(j-1)))
					temp2 = maxf64(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

				//              Load array V with the upper or lower triangular part
				//              of the matrix in band form.
			label1180:
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
				*srnamt = "DSBEVX"
				golapack.Dsbevx('V', 'A', uplo, &n, &kd, v, ldu, u, ldu, &vl, &vu, &il, &iu, &abstol, &m, wa2, z, ldu, work, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSBEVX(V,A,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1280
					}
				}

				//              Do tests 52 and 53 (or +54)
				Dsyt21(toPtr(1), uplo, &n, toPtr(0), a, ldu, wa2, d2, z, ldu, v, ldu, tau, work, result.Off(ntest-1))

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

				*srnamt = "DSBEVX_2STAGE"
				golapack.Dsbevx2stage('N', 'A', uplo, &n, &kd, v, ldu, u, ldu, &vl, &vu, &il, &iu, &abstol, &m3, wa3, z, ldu, work, lwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSBEVX_2STAGE(N,A,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label1280
					}
				}

				//              Do test 54 (or +54)
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, math.Abs(wa2.Get(j-1)), math.Abs(wa3.Get(j-1)))
					temp2 = maxf64(temp2, math.Abs(wa2.Get(j-1)-wa3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

			label1280:
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

				*srnamt = "DSBEVX"
				golapack.Dsbevx('V', 'I', uplo, &n, &kd, v, ldu, u, ldu, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, work, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSBEVX(V,I,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1370
					}
				}

				//              Do tests 55 and 56 (or +54)
				Dsyt22(toPtr(1), uplo, &n, &m2, toPtr(0), a, ldu, wa2, d2, z, ldu, v, ldu, tau, work, result.Off(ntest-1))

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

				*srnamt = "DSBEVX_2STAGE"
				golapack.Dsbevx2stage('N', 'I', uplo, &n, &kd, v, ldu, u, ldu, &vl, &vu, &il, &iu, &abstol, &m3, wa3, z, ldu, work, lwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSBEVX_2STAGE(N,I,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label1370
					}
				}

				//              Do test 57 (or +54)
				temp1 = Dsxt1(toPtr(1), wa2, &m2, wa3, &m3, &abstol, &ulp, &unfl)
				temp2 = Dsxt1(toPtr(1), wa3, &m3, wa2, &m2, &abstol, &ulp, &unfl)
				if n > 0 {
					temp3 = maxf64(math.Abs(wa1.Get(0)), math.Abs(wa1.Get(n-1)))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/maxf64(unfl, temp3*ulp))
				//
			label1370:
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

				*srnamt = "DSBEVX"
				golapack.Dsbevx('V', 'V', uplo, &n, &kd, v, ldu, u, ldu, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, work, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSBEVX(V,V,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1460
					}
				}

				//              Do tests 58 and 59 (or +54)
				Dsyt22(toPtr(1), uplo, &n, &m2, toPtr(0), a, ldu, wa2, d2, z, ldu, v, ldu, tau, work, result.Off(ntest-1))

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

				*srnamt = "DSBEVX_2STAGE"
				golapack.Dsbevx2stage('N', 'V', uplo, &n, &kd, v, ldu, u, ldu, &vl, &vu, &il, &iu, &abstol, &m3, wa3, z, ldu, work, lwork, iwork, toSlice(iwork, 5*n+1-1), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSBEVX_2STAGE(N,V,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label1460
					}
				}

				if m3 == 0 && n > 0 {
					result.Set(ntest-1, ulpinv)
					goto label1460
				}

				//              Do test 60 (or +54)
				temp1 = Dsxt1(toPtr(1), wa2, &m2, wa3, &m3, &abstol, &ulp, &unfl)
				temp2 = Dsxt1(toPtr(1), wa3, &m3, wa2, &m2, &abstol, &ulp, &unfl)
				if n > 0 {
					temp3 = maxf64(math.Abs(wa1.Get(0)), math.Abs(wa1.Get(n-1)))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/maxf64(unfl, temp3*ulp))

			label1460:
				;

				//              7)      Call DSYEVD
				golapack.Dlacpy(' ', &n, &n, a, lda, v, ldu)

				ntest = ntest + 1
				*srnamt = "DSYEVD"
				golapack.Dsyevd('V', uplo, &n, a, ldu, d1, work, &lwedc, iwork, &liwedc, &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSYEVD(V,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1480
					}
				}

				//              Do tests 61 and 62 (or +54)
				Dsyt21(toPtr(1), uplo, &n, toPtr(0), v, ldu, d1, d2, a, ldu, z, ldu, tau, work, result.Off(ntest-1))

				golapack.Dlacpy(' ', &n, &n, v, ldu, a, lda)

				ntest = ntest + 2
				*srnamt = "DSYEVD_2STAGE"
				golapack.Dsyevd2stage('N', uplo, &n, a, ldu, d3, work, lwork, iwork, &liwedc, &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSYEVD_2STAGE(N,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label1480
					}
				}

				//              Do test 63 (or +54)
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, math.Abs(d1.Get(j-1)), math.Abs(d3.Get(j-1)))
					temp2 = maxf64(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

			label1480:
				;

				//              8)      Call DSPEVD.
				golapack.Dlacpy(' ', &n, &n, v, ldu, a, lda)

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
				*srnamt = "DSPEVD"
				golapack.Dspevd('V', uplo, &n, work, d1, z, ldu, work.Off(indx-1), toPtr(lwedc-indx+1), iwork, &liwedc, &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSPEVD(V,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1580
					}
				}

				//              Do tests 64 and 65 (or +54)
				Dsyt21(toPtr(1), uplo, &n, toPtr(0), a, lda, d1, d2, z, ldu, v, ldu, tau, work, result.Off(ntest-1))

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
				*srnamt = "DSPEVD"
				golapack.Dspevd('N', uplo, &n, work, d3, z, ldu, work.Off(indx-1), toPtr(lwedc-indx+1), iwork, &liwedc, &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSPEVD(N,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label1580
					}
				}

				//              Do test 66 (or +54)
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, math.Abs(d1.Get(j-1)), math.Abs(d3.Get(j-1)))
					temp2 = maxf64(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))
			label1580:
				;

				//              9)      Call DSBEVD.
				if jtype <= 7 {
					kd = 1
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
				*srnamt = "DSBEVD"
				golapack.Dsbevd('V', uplo, &n, &kd, v, ldu, d1, z, ldu, work, &lwedc, iwork, &liwedc, &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSBEVD(V,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1680
					}
				}

				//              Do tests 67 and 68 (or +54)
				Dsyt21(toPtr(1), uplo, &n, toPtr(0), a, lda, d1, d2, z, ldu, v, ldu, tau, work, result.Off(ntest-1))

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
				*srnamt = "DSBEVD_2STAGE"
				golapack.Dsbevd2stage('N', uplo, &n, &kd, v, ldu, d3, z, ldu, work, lwork, iwork, &liwedc, &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSBEVD_2STAGE(N,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label1680
					}
				}

				//              Do test 69 (or +54)
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, math.Abs(d1.Get(j-1)), math.Abs(d3.Get(j-1)))
					temp2 = maxf64(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

			label1680:
				;

				golapack.Dlacpy(' ', &n, &n, a, lda, v, ldu)
				ntest = ntest + 1
				*srnamt = "DSYEVR"
				golapack.Dsyevr('V', 'A', uplo, &n, a, ldu, &vl, &vu, &il, &iu, &abstol, &m, wa1, z, ldu, iwork, work, lwork, toSlice(iwork, 2*n+1-1), toPtr((*liwork)-2*n), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSYEVR(V,A,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1700
					}
				}

				//              Do tests 70 and 71 (or ... )
				golapack.Dlacpy(' ', &n, &n, v, ldu, a, lda)

				Dsyt21(toPtr(1), uplo, &n, toPtr(0), a, ldu, wa1, d2, z, ldu, v, ldu, tau, work, result.Off(ntest-1))

				ntest = ntest + 2
				*srnamt = "DSYEVR_2STAGE"
				golapack.Dsyevr2stage('N', 'A', uplo, &n, a, ldu, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, iwork, work, lwork, toSlice(iwork, 2*n+1-1), toPtr((*liwork)-2*n), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSYEVR_2STAGE(N,A,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label1700
					}
				}

				//              Do test 72 (or ... )
				temp1 = zero
				temp2 = zero
				for j = 1; j <= n; j++ {
					temp1 = maxf64(temp1, math.Abs(wa1.Get(j-1)), math.Abs(wa2.Get(j-1)))
					temp2 = maxf64(temp2, math.Abs(wa1.Get(j-1)-wa2.Get(j-1)))
				}
				result.Set(ntest-1, temp2/maxf64(unfl, ulp*maxf64(temp1, temp2)))

			label1700:
				;

				ntest = ntest + 1
				golapack.Dlacpy(' ', &n, &n, v, ldu, a, lda)
				*srnamt = "DSYEVR"
				golapack.Dsyevr('V', 'I', uplo, &n, a, ldu, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, iwork, work, lwork, toSlice(iwork, 2*n+1-1), toPtr((*liwork)-2*n), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSYEVR(V,I,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1710
					}
				}

				//              Do tests 73 and 74 (or +54)
				golapack.Dlacpy(' ', &n, &n, v, ldu, a, lda)

				Dsyt22(toPtr(1), uplo, &n, &m2, toPtr(0), a, ldu, wa2, d2, z, ldu, v, ldu, tau, work, result.Off(ntest-1))

				ntest = ntest + 2
				golapack.Dlacpy(' ', &n, &n, v, ldu, a, lda)
				*srnamt = "DSYEVR_2STAGE"
				golapack.Dsyevr2stage('N', 'I', uplo, &n, a, ldu, &vl, &vu, &il, &iu, &abstol, &m3, wa3, z, ldu, iwork, work, lwork, toSlice(iwork, 2*n+1-1), toPtr((*liwork)-2*n), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSYEVR_2STAGE(N,I,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label1710
					}
				}

				//              Do test 75 (or +54)
				temp1 = Dsxt1(toPtr(1), wa2, &m2, wa3, &m3, &abstol, &ulp, &unfl)
				temp2 = Dsxt1(toPtr(1), wa3, &m3, wa2, &m2, &abstol, &ulp, &unfl)
				result.Set(ntest-1, (temp1+temp2)/maxf64(unfl, ulp*temp3))
			label1710:
				;

				ntest = ntest + 1
				golapack.Dlacpy(' ', &n, &n, v, ldu, a, lda)
				*srnamt = "DSYEVR"
				golapack.Dsyevr('V', 'V', uplo, &n, a, ldu, &vl, &vu, &il, &iu, &abstol, &m2, wa2, z, ldu, iwork, work, lwork, toSlice(iwork, 2*n+1-1), toPtr((*liwork)-2*n), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSYEVR(V,V,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest+1-1, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label700
					}
				}

				//              Do tests 76 and 77 (or +54)
				golapack.Dlacpy(' ', &n, &n, v, ldu, a, lda)

				Dsyt22(toPtr(1), uplo, &n, &m2, toPtr(0), a, ldu, wa2, d2, z, ldu, v, ldu, tau, work, result.Off(ntest-1))

				ntest = ntest + 2
				golapack.Dlacpy(' ', &n, &n, v, ldu, a, lda)
				*srnamt = "DSYEVR_2STAGE"
				golapack.Dsyevr2stage('N', 'V', uplo, &n, a, ldu, &vl, &vu, &il, &iu, &abstol, &m3, wa3, z, ldu, iwork, work, lwork, toSlice(iwork, 2*n+1-1), toPtr((*liwork)-2*n), &iinfo)
				if iinfo != 0 {
					fmt.Printf(" DDRVST2STG: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, ISEED=%5d\n", "DSYEVR_2STAGE(N,V,"+string(uplo)+")", iinfo, n, jtype, ioldsd)
					*info = int(math.Abs(float64(iinfo)))
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label700
					}
				}

				if m3 == 0 && n > 0 {
					result.Set(ntest-1, ulpinv)
					goto label700
				}

				//              Do test 78 (or +54)
				temp1 = Dsxt1(toPtr(1), wa2, &m2, wa3, &m3, &abstol, &ulp, &unfl)
				temp2 = Dsxt1(toPtr(1), wa3, &m3, wa2, &m2, &abstol, &ulp, &unfl)
				if n > 0 {
					temp3 = maxf64(math.Abs(wa1.Get(0)), math.Abs(wa1.Get(n-1)))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/maxf64(unfl, temp3*ulp))

				golapack.Dlacpy(' ', &n, &n, v, ldu, a, lda)

			}

			//           End of Loop -- Check for RESULT(j) > THRESH
			ntestt = ntestt + ntest

			Dlafts([]byte("DST"), &n, &n, &jtype, &ntest, result, &ioldsd, thresh, &nerrs, t)

		label1730:
		}
	}

	//     Summary
	Alasvm([]byte("DST"), &nerrs, &ntestt, toPtr(0))
}
