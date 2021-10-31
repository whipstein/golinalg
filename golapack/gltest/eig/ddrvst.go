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

// ddrvst checks the symmetric eigenvalue problem drivers.
//
//              Dstev computes all eigenvalues and, optionally,
//              eigenvectors of a real symmetric tridiagonal matrix.
//
//              Dstevx computes selected eigenvalues and, optionally,
//              eigenvectors of a real symmetric tridiagonal matrix.
//
//              Dstevr computes selected eigenvalues and, optionally,
//              eigenvectors of a real symmetric tridiagonal matrix
//              using the Relatively Robust Representation where it can.
//
//              Dsyev computes all eigenvalues and, optionally,
//              eigenvectors of a real symmetric matrix.
//
//              Dsyevx computes selected eigenvalues and, optionally,
//              eigenvectors of a real symmetric matrix.
//
//              Dsyevr computes selected eigenvalues and, optionally,
//              eigenvectors of a real symmetric matrix
//              using the Relatively Robust Representation where it can.
//
//              Dspev computes all eigenvalues and, optionally,
//              eigenvectors of a real symmetric matrix in packed
//              storage.
//
//              Dspevx computes selected eigenvalues and, optionally,
//              eigenvectors of a real symmetric matrix in packed
//              storage.
//
//              Dsbev computes all eigenvalues and, optionally,
//              eigenvectors of a real symmetric band matrix.
//
//              Dsbevx computes selected eigenvalues and, optionally,
//              eigenvectors of a real symmetric band matrix.
//
//              Dsyevd computes all eigenvalues and, optionally,
//              eigenvectors of a real symmetric matrix using
//              a divide and conquer algorithm.
//
//              Dspevd computes all eigenvalues and, optionally,
//              eigenvectors of a real symmetric matrix in packed
//              storage, using a divide and conquer algorithm.
//
//              Dsbevd computes all eigenvalues and, optionally,
//              eigenvectors of a real symmetric band matrix,
//              using a divide and conquer algorithm.
//
//      When ddrvst is called, a number of matrix "sizes" ("n's") and a
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
func ddrvst(nsizes int, nn []int, ntypes int, dotype []bool, iseed []int, thresh float64, nounit int, a *mat.Matrix, d1, d2, d3, d4, eveigs, wa1, wa2, wa3 *mat.Vector, u, v *mat.Matrix, tau *mat.Vector, z *mat.Matrix, work *mat.Vector, lwork int, iwork []int, liwork int, result *mat.Vector, t *testing.T) (nfails, ntestt int, err error) {
	var badnn bool
	var uplo mat.MatUplo
	var abstol, aninv, anorm, cond, half, one, ovfl, rtovfl, rtunfl, temp1, temp2, temp3, ten, two, ulp, ulpinv, unfl, vl, vu, zero float64
	var i, idiag, ihbw, iinfo, il, imode, indx, irow, itemp, itype, iu, j, j1, j2, jcol, jsize, jtype, kd, lgn, liwedc, lwedc, m2, m3, maxtyp, mtypes, n, nerrs, nmats, nmax, ntest int
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

	badnn = false
	nmax = 1
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
	} else if a.Rows < nmax {
		err = fmt.Errorf("a.Rows < nmax: a.Rows=%v, nmax=%v", a.Rows, nmax)
	} else if u.Rows < nmax {
		err = fmt.Errorf("u.Rows < nmax: u.Rows=%v, nmax=%v", u.Rows, nmax)
	} else if 2*pow(max(2, nmax), 2) > lwork {
		err = fmt.Errorf("2*pow(max(2, nmax), 2) > lwork: nmax=%v, lwork=%v", nmax, lwork)
	}

	if err != nil {
		gltest.Xerbla2("ddrvst", err)
		return
	}

	//     Quick return if nothing to do
	if nsizes == 0 || ntypes == 0 {
		return
	}

	//     More Important constants
	unfl = golapack.Dlamch(SafeMinimum)
	ovfl = golapack.Dlamch(Overflow)
	unfl, ovfl = golapack.Dlabad(unfl, ovfl)
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)
	ulpinv = one / ulp
	rtunfl = math.Sqrt(unfl)
	rtovfl = math.Sqrt(ovfl)

	//     Loop over sizes, types
	for i = 1; i <= 4; i++ {
		iseed2[i-1] = iseed[i-1]
		iseed3[i-1] = iseed[i-1]
	}

	nerrs = 0
	nfails = 0
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
			//c           LIWEDC = 6 + 6*N + 5*N*LGN
			liwedc = 3 + 5*n
		} else {
			lwedc = 9
			//c           LIWEDC = 12
			liwedc = 8
		}
		aninv = one / float64(max(1, n))

		if nsizes != 1 {
			mtypes = min(maxtyp, ntypes)
		} else {
			mtypes = min(maxtyp+1, ntypes)
		}

		for jtype = 1; jtype <= mtypes; jtype++ {

			if !dotype[jtype-1] {
				goto label1730
			}
			nmats = nmats + 1
			ntest = 0

			for j = 1; j <= 4; j++ {
				ioldsd[j-1] = iseed[j-1]
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

			golapack.Dlaset(Full, a.Rows, n, zero, zero, a)
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
				iinfo, err = matgen.Dlatms(n, n, 'S', &iseed, 'S', work, imode, cond, anorm, 0, 0, 'N', a, work.Off(n))

			} else if itype == 5 {
				//              Symmetric, eigenvalues specified
				iinfo, err = matgen.Dlatms(n, n, 'S', &iseed, 'S', work, imode, cond, anorm, n, n, 'N', a, work.Off(n))

			} else if itype == 7 {
				//              Diagonal, random eigenvalues
				idumma[0] = 1
				iinfo, err = matgen.Dlatmr(n, n, 'S', &iseed, 'S', work, 6, one, one, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, 0, 0, zero, anorm, 'N', a, &iwork)

			} else if itype == 8 {
				//              Symmetric, random eigenvalues
				idumma[0] = 1
				iinfo, err = matgen.Dlatmr(n, n, 'S', &iseed, 'S', work, 6, one, one, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, n, n, zero, anorm, 'N', a, &iwork)

			} else if itype == 9 {
				//              Symmetric banded, eigenvalues specified
				ihbw = int(float64(n-1) * matgen.Dlarnd(1, &iseed3))
				iinfo, err = matgen.Dlatms(n, n, 'S', &iseed, 'S', work, imode, cond, anorm, ihbw, ihbw, 'Z', u, work.Off(n))

				//              Store as dense matrix for most routines.
				golapack.Dlaset(Full, a.Rows, n, zero, zero, a)
				for idiag = -ihbw; idiag <= ihbw; idiag++ {
					irow = ihbw - idiag + 1
					j1 = max(1, idiag+1)
					j2 = min(n, n+idiag)
					for j = j1; j <= j2; j++ {
						i = j - idiag
						a.Set(i-1, j-1, u.Get(irow-1, j-1))
					}
				}
			} else {
				iinfo = 1
			}

			if err != nil || iinfo != 0 {
				t.Fail()
				nerrs++
				fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Generator", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				return
			}

		label110:
			;

			abstol = unfl + unfl
			if n <= 1 {
				il = 1
				iu = n
			} else {
				il = 1 + (n-1)*int(matgen.Dlarnd(1, &iseed2))
				iu = 1 + (n-1)*int(matgen.Dlarnd(1, &iseed2))
				if il > iu {
					itemp = il
					il = iu
					iu = itemp
				}
			}

			//           3)      If matrix is tridiagonal, call Dstev and Dstevx.
			if jtype <= 7 {
				ntest = 1
				for i = 1; i <= n; i++ {
					d1.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d2.Set(i-1, float64(a.Get(i, i-1)))
				}
				*srnamt = "Dstev"
				if iinfo, err = golapack.Dstev('V', n, d1, d2, z, work); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstev(V)", iinfo, n, jtype, ioldsd)
					if err != nil {
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
					d4.Set(i-1, float64(a.Get(i, i-1)))
				}
				dstt21(n, 0, d3, d4, d1, d2, z, work, result)

				ntest = 3
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i, i-1)))
				}
				*srnamt = "Dstev"
				if iinfo, err = golapack.Dstev('N', n, d3, d4, z, work); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstev(N)", iinfo, n, jtype, ioldsd)
					if err != nil {
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
					temp1 = math.Max(temp1, math.Max(math.Abs(d1.Get(j-1)), math.Abs(d3.Get(j-1))))
					temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(2, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

			label180:
				;

				ntest = 4
				for i = 1; i <= n; i++ {
					eveigs.Set(i-1, d3.Get(i-1))
					d1.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d2.Set(i-1, float64(a.Get(i, i-1)))
				}
				*srnamt = "Dstevx"
				if _, iinfo, err = golapack.Dstevx('V', 'A', n, d1, d2, vl, vu, il, iu, abstol, wa1, z, work, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstevx(V,A)", iinfo, n, jtype, ioldsd)
					if err != nil {
						return
					} else {
						result.Set(3, ulpinv)
						result.Set(4, ulpinv)
						result.Set(5, ulpinv)
						goto label250
					}
				}
				if n > 0 {
					temp3 = math.Max(math.Abs(wa1.Get(0)), math.Abs(wa1.Get(n-1)))
				} else {
					temp3 = zero
				}

				//              Do tests 4 and 5.
				for i = 1; i <= n; i++ {
					d3.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i, i-1)))
				}
				dstt21(n, 0, d3, d4, wa1, d2, z, work, result.Off(3))

				ntest = 6
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i, i-1)))
				}
				*srnamt = "Dstevx"
				if m2, iinfo, err = golapack.Dstevx('N', 'A', n, d3, d4, vl, vu, il, iu, abstol, wa2, z, work, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstevx(N,A)", iinfo, n, jtype, ioldsd)
					if err != nil {
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
					temp1 = math.Max(temp1, math.Max(math.Abs(wa2.Get(j-1)), math.Abs(eveigs.Get(j-1))))
					temp2 = math.Max(temp2, math.Abs(wa2.Get(j-1)-eveigs.Get(j-1)))
				}
				result.Set(5, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

			label250:
				;

				ntest = 7
				for i = 1; i <= n; i++ {
					d1.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d2.Set(i-1, float64(a.Get(i, i-1)))
				}
				*srnamt = "Dstevr"
				if _, iinfo, err = golapack.Dstevr('V', 'A', n, d1, d2, vl, vu, il, iu, abstol, wa1, z, &iwork, work, lwork, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstevr(V,A)", iinfo, n, jtype, ioldsd)
					if err != nil {
						return
					} else {
						result.Set(6, ulpinv)
						result.Set(7, ulpinv)
						goto label320
					}
				}
				if n > 0 {
					temp3 = math.Max(math.Abs(wa1.Get(0)), math.Abs(wa1.Get(n-1)))
				} else {
					temp3 = zero
				}

				//              Do tests 7 and 8.
				for i = 1; i <= n; i++ {
					d3.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i, i-1)))
				}
				dstt21(n, 0, d3, d4, wa1, d2, z, work, result.Off(6))

				ntest = 9
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i, i-1)))
				}
				*srnamt = "Dstevr"
				if m2, iinfo, err = golapack.Dstevr('N', 'A', n, d3, d4, vl, vu, il, iu, abstol, wa2, z, &iwork, work, lwork, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstevr(N,A)", iinfo, n, jtype, ioldsd)
					if err != nil {
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
					temp1 = math.Max(temp1, math.Max(math.Abs(wa2.Get(j-1)), math.Abs(eveigs.Get(j-1))))
					temp2 = math.Max(temp2, math.Abs(wa2.Get(j-1)-eveigs.Get(j-1)))
				}
				result.Set(8, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

			label320:
				;

				ntest = 10
				for i = 1; i <= n; i++ {
					d1.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d2.Set(i-1, float64(a.Get(i, i-1)))
				}
				*srnamt = "Dstevx"
				if m2, iinfo, err = golapack.Dstevx('V', 'I', n, d1, d2, vl, vu, il, iu, abstol, wa2, z, work, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstevx(V,I)", iinfo, n, jtype, ioldsd)
					if err != nil {
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
					d4.Set(i-1, float64(a.Get(i, i-1)))
				}
				dstt22(n, m2, 0, d3, d4, wa2, d2, z, work.Matrix(max(1, m2), opts), result.Off(9))

				ntest = 12
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i, i-1)))
				}
				*srnamt = "Dstevx"
				if m3, iinfo, err = golapack.Dstevx('N', 'I', n, d3, d4, vl, vu, il, iu, abstol, wa3, z, work, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstevx(N,I)", iinfo, n, jtype, ioldsd)
					if err != nil {
						return
					} else {
						result.Set(11, ulpinv)
						goto label380
					}
				}

				//              Do test 12.
				temp1 = dsxt1(1, wa2, m2, wa3, m3, abstol, ulp, unfl)
				temp2 = dsxt1(1, wa3, m3, wa2, m2, abstol, ulp, unfl)
				result.Set(11, (temp1+temp2)/math.Max(unfl, ulp*temp3))

			label380:
				;

				ntest = 12
				if n > 0 {
					if il != 1 {
						vl = wa1.Get(il-1) - math.Max(half*(wa1.Get(il-1)-wa1.Get(il-1-1)), math.Max(ten*ulp*temp3, ten*rtunfl))
					} else {
						vl = wa1.Get(0) - math.Max(half*(wa1.Get(n-1)-wa1.Get(0)), math.Max(ten*ulp*temp3, ten*rtunfl))
					}
					if iu != n {
						vu = wa1.Get(iu-1) + math.Max(half*(wa1.Get(iu)-wa1.Get(iu-1)), math.Max(ten*ulp*temp3, ten*rtunfl))
					} else {
						vu = wa1.Get(n-1) + math.Max(half*(wa1.Get(n-1)-wa1.Get(0)), math.Max(ten*ulp*temp3, ten*rtunfl))
					}
				} else {
					vl = zero
					vu = one
				}

				for i = 1; i <= n; i++ {
					d1.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d2.Set(i-1, float64(a.Get(i, i-1)))
				}
				*srnamt = "Dstevx"
				if m2, iinfo, err = golapack.Dstevx('V', 'V', n, d1, d2, vl, vu, il, iu, abstol, wa2, z, work, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstevx(V,V)", iinfo, n, jtype, ioldsd)
					if err != nil {
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
					d4.Set(i-1, float64(a.Get(i, i-1)))
				}
				dstt22(n, m2, 0, d3, d4, wa2, d2, z, work.Matrix(max(1, m2), opts), result.Off(12))

				ntest = 15
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i, i-1)))
				}
				*srnamt = "Dstevx"
				if m3, iinfo, err = golapack.Dstevx('N', 'V', n, d3, d4, vl, vu, il, iu, abstol, wa3, z, work, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstevx(N,V)", iinfo, n, jtype, ioldsd)
					if err != nil {
						return
					} else {
						result.Set(14, ulpinv)
						goto label440
					}
				}

				//              Do test 15.
				temp1 = dsxt1(1, wa2, m2, wa3, m3, abstol, ulp, unfl)
				temp2 = dsxt1(1, wa3, m3, wa2, m2, abstol, ulp, unfl)
				result.Set(14, (temp1+temp2)/math.Max(unfl, temp3*ulp))

			label440:
				;

				ntest = 16
				for i = 1; i <= n; i++ {
					d1.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d2.Set(i-1, float64(a.Get(i, i-1)))
				}
				*srnamt = "Dstevd"
				if iinfo, err = golapack.Dstevd('V', n, d1, d2, z, work, lwedc, &iwork, liwedc); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstevd(V)", iinfo, n, jtype, ioldsd)
					if err != nil {
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
					d4.Set(i-1, float64(a.Get(i, i-1)))
				}
				dstt21(n, 0, d3, d4, d1, d2, z, work, result.Off(15))

				ntest = 18
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i, i-1)))
				}
				*srnamt = "Dstevd"
				if iinfo, err = golapack.Dstevd('N', n, d3, d4, z, work, lwedc, &iwork, liwedc); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstevd(N)", iinfo, n, jtype, ioldsd)
					if err != nil {
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
					temp1 = math.Max(temp1, math.Max(math.Abs(eveigs.Get(j-1)), math.Abs(d3.Get(j-1))))
					temp2 = math.Max(temp2, math.Abs(eveigs.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(17, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

			label510:
				;

				ntest = 19
				for i = 1; i <= n; i++ {
					d1.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d2.Set(i-1, float64(a.Get(i, i-1)))
				}
				*srnamt = "Dstevr"
				if m2, iinfo, err = golapack.Dstevr('V', 'I', n, d1, d2, vl, vu, il, iu, abstol, wa2, z, &iwork, work, lwork, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstevr(V,I)", iinfo, n, jtype, ioldsd)
					if err != nil {
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
					d4.Set(i-1, float64(a.Get(i, i-1)))
				}
				dstt22(n, m2, 0, d3, d4, wa2, d2, z, work.Matrix(max(1, m2), opts), result.Off(18))

				ntest = 21
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i, i-1)))
				}
				*srnamt = "Dstevr"
				if m3, iinfo, err = golapack.Dstevr('N', 'I', n, d3, d4, vl, vu, il, iu, abstol, wa3, z, &iwork, work, lwork, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstevr(N,I)", iinfo, n, jtype, ioldsd)
					if err != nil {
						return
					} else {
						result.Set(20, ulpinv)
						goto label570
					}
				}

				//              Do test 21.
				temp1 = dsxt1(1, wa2, m2, wa3, m3, abstol, ulp, unfl)
				temp2 = dsxt1(1, wa3, m3, wa2, m2, abstol, ulp, unfl)
				result.Set(20, (temp1+temp2)/math.Max(unfl, ulp*temp3))

			label570:
				;

				ntest = 21
				if n > 0 {
					if il != 1 {
						vl = wa1.Get(il-1) - math.Max(half*(wa1.Get(il-1)-wa1.Get(il-1-1)), math.Max(ten*ulp*temp3, ten*rtunfl))
					} else {
						vl = wa1.Get(0) - math.Max(half*(wa1.Get(n-1)-wa1.Get(0)), math.Max(ten*ulp*temp3, ten*rtunfl))
					}
					if iu != n {
						vu = wa1.Get(iu-1) + math.Max(half*(wa1.Get(iu)-wa1.Get(iu-1)), math.Max(ten*ulp*temp3, ten*rtunfl))
					} else {
						vu = wa1.Get(n-1) + math.Max(half*(wa1.Get(n-1)-wa1.Get(0)), math.Max(ten*ulp*temp3, ten*rtunfl))
					}
				} else {
					vl = zero
					vu = one
				}

				for i = 1; i <= n; i++ {
					d1.Set(i-1, float64(a.Get(i-1, i-1)))
				}
				for i = 1; i <= n-1; i++ {
					d2.Set(i-1, float64(a.Get(i, i-1)))
				}
				*srnamt = "Dstevr"
				if m2, iinfo, err = golapack.Dstevr('V', 'V', n, d1, d2, vl, vu, il, iu, abstol, wa2, z, &iwork, work, lwork, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstevr(V,V)", iinfo, n, jtype, ioldsd)
					if err != nil {
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
					d4.Set(i-1, float64(a.Get(i, i-1)))
				}
				dstt22(n, m2, 0, d3, d4, wa2, d2, z, work.Matrix(max(1, m2), opts), result.Off(21))

				ntest = 24
				for i = 1; i <= n-1; i++ {
					d4.Set(i-1, float64(a.Get(i, i-1)))
				}
				*srnamt = "Dstevr"
				if m3, iinfo, err = golapack.Dstevr('N', 'V', n, d3, d4, vl, vu, il, iu, abstol, wa3, z, &iwork, work, lwork, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dstevr(N,V)", iinfo, n, jtype, ioldsd)
					if err != nil {
						return
					} else {
						result.Set(23, ulpinv)
						goto label630
					}
				}

				//              Do test 24.
				temp1 = dsxt1(1, wa2, m2, wa3, m3, abstol, ulp, unfl)
				temp2 = dsxt1(1, wa3, m3, wa2, m2, abstol, ulp, unfl)
				result.Set(23, (temp1+temp2)/math.Max(unfl, temp3*ulp))

			label630:
			} else {

				for i = 1; i <= 24; i++ {
					result.Set(i-1, zero)
				}
				ntest = 24
			}

			//           Perform remaining tests storing upper or lower triangular
			//           part of matrix.
			for _, uplo = range mat.IterMatUplo(false) {

				//              4)      Call Dsyev and Dsyevx.
				golapack.Dlacpy(Full, n, n, a, v)

				ntest = ntest + 1
				*srnamt = "Dsyev"
				if iinfo, err = golapack.Dsyev('V', uplo, n, a, d1, work, lwork); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsyev(V,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label660
					}
				}

				//              Do tests 25 and 26 (or +54)
				dsyt21(1, uplo, n, 0, v, d1, d2, a, z, tau, work, result.Off(ntest-1))

				golapack.Dlacpy(Full, n, n, v, a)

				ntest = ntest + 2
				*srnamt = "Dsyev"
				if iinfo, err = golapack.Dsyev('N', uplo, n, a, d3, work, lwork); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsyev(N,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
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
					temp1 = math.Max(temp1, math.Max(math.Abs(d1.Get(j-1)), math.Abs(d3.Get(j-1))))
					temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

			label660:
				;
				golapack.Dlacpy(Full, n, n, v, a)

				ntest = ntest + 1

				if n > 0 {
					temp3 = math.Max(math.Abs(d1.Get(0)), math.Abs(d1.Get(n-1)))
					if il != 1 {
						vl = d1.Get(il-1) - math.Max(half*(d1.Get(il-1)-d1.Get(il-1-1)), math.Max(ten*ulp*temp3, ten*rtunfl))
					} else if n > 0 {
						vl = d1.Get(0) - math.Max(half*(d1.Get(n-1)-d1.Get(0)), math.Max(ten*ulp*temp3, ten*rtunfl))
					}
					if iu != n {
						vu = d1.Get(iu-1) + math.Max(half*(d1.Get(iu)-d1.Get(iu-1)), math.Max(ten*ulp*temp3, ten*rtunfl))
					} else if n > 0 {
						vu = d1.Get(n-1) + math.Max(half*(d1.Get(n-1)-d1.Get(0)), math.Max(ten*ulp*temp3, ten*rtunfl))
					}
				} else {
					temp3 = zero
					vl = zero
					vu = one
				}

				*srnamt = "Dsyevx"
				if _, iinfo, err = golapack.Dsyevx('V', 'A', uplo, n, a, vl, vu, il, iu, abstol, wa1, z, work, lwork, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsyevx(V,A,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label680
					}
				}

				//              Do tests 28 and 29 (or +54)
				golapack.Dlacpy(Full, n, n, v, a)

				dsyt21(1, uplo, n, 0, a, d1, d2, z, v, tau, work, result.Off(ntest-1))

				ntest = ntest + 2
				*srnamt = "Dsyevx"
				if m2, iinfo, err = golapack.Dsyevx('N', 'A', uplo, n, a, vl, vu, il, iu, abstol, wa2, z, work, lwork, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsyevx(N,A,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					if err != nil {
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
					temp1 = math.Max(temp1, math.Max(math.Abs(wa1.Get(j-1)), math.Abs(wa2.Get(j-1))))
					temp2 = math.Max(temp2, math.Abs(wa1.Get(j-1)-wa2.Get(j-1)))
				}
				result.Set(ntest-1, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

			label680:
				;

				ntest = ntest + 1
				golapack.Dlacpy(Full, n, n, v, a)
				*srnamt = "Dsyevx"
				if m2, iinfo, err = golapack.Dsyevx('V', 'I', uplo, n, a, vl, vu, il, iu, abstol, wa2, z, work, lwork, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsyevx(V,I,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label690
					}
				}

				//              Do tests 31 and 32 (or +54)
				golapack.Dlacpy(Full, n, n, v, a)

				dsyt22(1, uplo, n, m2, 0, a, wa2, d2, z, v, tau, work, result.Off(ntest-1))

				ntest = ntest + 2
				golapack.Dlacpy(Full, n, n, v, a)
				*srnamt = "Dsyevx"
				if m3, iinfo, err = golapack.Dsyevx('N', 'I', uplo, n, a, vl, vu, il, iu, abstol, wa3, z, work, lwork, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsyevx(N,I,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					if err != nil {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label690
					}
				}

				//              Do test 33 (or +54)
				temp1 = dsxt1(1, wa2, m2, wa3, m3, abstol, ulp, unfl)
				temp2 = dsxt1(1, wa3, m3, wa2, m2, abstol, ulp, unfl)
				result.Set(ntest-1, (temp1+temp2)/math.Max(unfl, ulp*temp3))
			label690:
				;

				ntest = ntest + 1
				golapack.Dlacpy(Full, n, n, v, a)
				*srnamt = "Dsyevx"
				if m2, iinfo, err = golapack.Dsyevx('V', 'V', uplo, n, a, vl, vu, il, iu, abstol, wa2, z, work, lwork, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsyevx(V,V,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					if err != nil {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label700
					}
				}

				//              Do tests 34 and 35 (or +54)
				golapack.Dlacpy(Full, n, n, v, a)

				dsyt22(1, uplo, n, m2, 0, a, wa2, d2, z, v, tau, work, result.Off(ntest-1))

				ntest = ntest + 2
				golapack.Dlacpy(Full, n, n, v, a)
				*srnamt = "Dsyevx"
				if m3, iinfo, err = golapack.Dsyevx('N', 'V', uplo, n, a, vl, vu, il, iu, abstol, wa3, z, work, lwork, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsyevx(N,V,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					if err != nil {
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
				temp1 = dsxt1(1, wa2, m2, wa3, m3, abstol, ulp, unfl)
				temp2 = dsxt1(1, wa3, m3, wa2, m2, abstol, ulp, unfl)
				if n > 0 {
					temp3 = math.Max(math.Abs(wa1.Get(0)), math.Abs(wa1.Get(n-1)))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/math.Max(unfl, temp3*ulp))

			label700:
				;

				//              5)      Call Dspev and Dspevx.
				golapack.Dlacpy(Full, n, n, v, a)

				//              Load array WORK with the upper or lower triangular
				//              part of the matrix in packed form.
				if uplo == Upper {
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
				*srnamt = "Dspev"
				if iinfo, err = golapack.Dspev('V', uplo, n, work, d1, z, v.VectorIdx(0)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dspev(V,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label800
					}
				}

				//              Do tests 37 and 38 (or +54)
				dsyt21(1, uplo, n, 0, a, d1, d2, z, v, tau, work, result.Off(ntest-1))

				if uplo == Upper {
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
				*srnamt = "Dspev"
				if iinfo, err = golapack.Dspev('N', uplo, n, work, d3, z, v.VectorIdx(0)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dspev(N,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
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
					temp1 = math.Max(temp1, math.Max(math.Abs(d1.Get(j-1)), math.Abs(d3.Get(j-1))))
					temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

				//              Load array WORK with the upper or lower triangular part
				//              of the matrix in packed form.
			label800:
				;
				if uplo == Upper {
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
					temp3 = math.Max(math.Abs(d1.Get(0)), math.Abs(d1.Get(n-1)))
					if il != 1 {
						vl = d1.Get(il-1) - math.Max(half*(d1.Get(il-1)-d1.Get(il-1-1)), math.Max(ten*ulp*temp3, ten*rtunfl))
					} else if n > 0 {
						vl = d1.Get(0) - math.Max(half*(d1.Get(n-1)-d1.Get(0)), math.Max(ten*ulp*temp3, ten*rtunfl))
					}
					if iu != n {
						vu = d1.Get(iu-1) + math.Max(half*(d1.Get(iu)-d1.Get(iu-1)), math.Max(ten*ulp*temp3, ten*rtunfl))
					} else if n > 0 {
						vu = d1.Get(n-1) + math.Max(half*(d1.Get(n-1)-d1.Get(0)), math.Max(ten*ulp*temp3, ten*rtunfl))
					}
				} else {
					temp3 = zero
					vl = zero
					vu = one
				}

				*srnamt = "Dspevx"
				if _, iinfo, err = golapack.Dspevx('V', 'A', uplo, n, work, vl, vu, il, iu, abstol, wa1, z, v.VectorIdx(0), &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dspevx(V,A,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label900
					}
				}

				//              Do tests 40 and 41 (or +54)
				dsyt21(1, uplo, n, 0, a, wa1, d2, z, v, tau, work, result.Off(ntest-1))

				ntest = ntest + 2

				if uplo == Upper {
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

				*srnamt = "Dspevx"
				if m2, iinfo, err = golapack.Dspevx('N', 'A', uplo, n, work, vl, vu, il, iu, abstol, wa2, z, v.VectorIdx(0), &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dspevx(N,A,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
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
					temp1 = math.Max(temp1, math.Max(math.Abs(wa1.Get(j-1)), math.Abs(wa2.Get(j-1))))
					temp2 = math.Max(temp2, math.Abs(wa1.Get(j-1)-wa2.Get(j-1)))
				}
				result.Set(ntest-1, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

			label900:
				;
				if uplo == Upper {
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

				*srnamt = "Dspevx"
				if m2, iinfo, err = golapack.Dspevx('V', 'I', uplo, n, work, vl, vu, il, iu, abstol, wa2, z, v.VectorIdx(0), &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dspevx(V,I,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label990
					}
				}

				//              Do tests 43 and 44 (or +54)
				dsyt22(1, uplo, n, m2, 0, a, wa2, d2, z, v, tau, work, result.Off(ntest-1))

				ntest = ntest + 2

				if uplo == Upper {
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

				*srnamt = "Dspevx"
				if m3, iinfo, err = golapack.Dspevx('N', 'I', uplo, n, work, vl, vu, il, iu, abstol, wa3, z, v.VectorIdx(0), &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dspevx(N,I,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
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
				temp1 = dsxt1(1, wa2, m2, wa3, m3, abstol, ulp, unfl)
				temp2 = dsxt1(1, wa3, m3, wa2, m2, abstol, ulp, unfl)
				if n > 0 {
					temp3 = math.Max(math.Abs(wa1.Get(0)), math.Abs(wa1.Get(n-1)))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/math.Max(unfl, temp3*ulp))

			label990:
				;
				if uplo == Upper {
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

				*srnamt = "Dspevx"
				if m2, iinfo, err = golapack.Dspevx('V', 'V', uplo, n, work, vl, vu, il, iu, abstol, wa2, z, v.VectorIdx(0), &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dspevx(V,V,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1080
					}
				}

				//              Do tests 46 and 47 (or +54)
				dsyt22(1, uplo, n, m2, 0, a, wa2, d2, z, v, tau, work, result.Off(ntest-1))

				ntest = ntest + 2

				if uplo == Upper {
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

				*srnamt = "Dspevx"
				if m3, iinfo, err = golapack.Dspevx('N', 'V', uplo, n, work, vl, vu, il, iu, abstol, wa3, z, v.VectorIdx(0), &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dspevx(N,V,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
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
				temp1 = dsxt1(1, wa2, m2, wa3, m3, abstol, ulp, unfl)
				temp2 = dsxt1(1, wa3, m3, wa2, m2, abstol, ulp, unfl)
				if n > 0 {
					temp3 = math.Max(math.Abs(wa1.Get(0)), math.Abs(wa1.Get(n-1)))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/math.Max(unfl, temp3*ulp))

			label1080:
				;

				//              6)      Call Dsbev and Dsbevx.
				if jtype <= 7 {
					kd = 1
				} else if jtype >= 8 && jtype <= 15 {
					kd = max(n-1, 0)
				} else {
					kd = ihbw
				}

				//              Load array V with the upper or lower triangular part
				//              of the matrix in band form.
				if uplo == Upper {
					for j = 1; j <= n; j++ {
						for i = max(1, j-kd); i <= j; i++ {
							v.Set(kd+1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				} else {
					for j = 1; j <= n; j++ {
						for i = j; i <= min(n, j+kd); i++ {
							v.Set(1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				}

				ntest = ntest + 1
				*srnamt = "Dsbev"
				if iinfo, err = golapack.Dsbev('V', uplo, n, kd, v, d1, z, work); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsbev(V,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1180
					}
				}

				//              Do tests 49 and 50 (or ... )
				dsyt21(1, uplo, n, 0, a, d1, d2, z, v, tau, work, result.Off(ntest-1))

				if uplo == Upper {
					for j = 1; j <= n; j++ {
						for i = max(1, j-kd); i <= j; i++ {
							v.Set(kd+1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				} else {
					for j = 1; j <= n; j++ {
						for i = j; i <= min(n, j+kd); i++ {
							v.Set(1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				}

				ntest = ntest + 2
				*srnamt = "Dsbev"
				if iinfo, err = golapack.Dsbev('N', uplo, n, kd, v, d3, z, work); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsbev(N,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
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
					temp1 = math.Max(temp1, math.Max(math.Abs(d1.Get(j-1)), math.Abs(d3.Get(j-1))))
					temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

				//              Load array V with the upper or lower triangular part
				//              of the matrix in band form.
			label1180:
				;
				if uplo == Upper {
					for j = 1; j <= n; j++ {
						for i = max(1, j-kd); i <= j; i++ {
							v.Set(kd+1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				} else {
					for j = 1; j <= n; j++ {
						for i = j; i <= min(n, j+kd); i++ {
							v.Set(1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				}

				ntest = ntest + 1
				*srnamt = "Dsbevx"
				if _, iinfo, err = golapack.Dsbevx('V', 'A', uplo, n, kd, v, u, vl, vu, il, iu, abstol, wa2, z, work, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsbevx(V,A,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1280
					}
				}

				//              Do tests 52 and 53 (or +54)
				dsyt21(1, uplo, n, 0, a, wa2, d2, z, v, tau, work, result.Off(ntest-1))

				ntest = ntest + 2

				if uplo == Upper {
					for j = 1; j <= n; j++ {
						for i = max(1, j-kd); i <= j; i++ {
							v.Set(kd+1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				} else {
					for j = 1; j <= n; j++ {
						for i = j; i <= min(n, j+kd); i++ {
							v.Set(1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				}

				*srnamt = "Dsbevx"
				if m3, iinfo, err = golapack.Dsbevx('N', 'A', uplo, n, kd, v, u, vl, vu, il, iu, abstol, wa3, z, work, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsbevx(N,A,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
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
					temp1 = math.Max(temp1, math.Max(math.Abs(wa2.Get(j-1)), math.Abs(wa3.Get(j-1))))
					temp2 = math.Max(temp2, math.Abs(wa2.Get(j-1)-wa3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

			label1280:
				;
				ntest = ntest + 1
				if uplo == Upper {
					for j = 1; j <= n; j++ {
						for i = max(1, j-kd); i <= j; i++ {
							v.Set(kd+1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				} else {
					for j = 1; j <= n; j++ {
						for i = j; i <= min(n, j+kd); i++ {
							v.Set(1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				}

				*srnamt = "Dsbevx"
				if m2, iinfo, err = golapack.Dsbevx('V', 'I', uplo, n, kd, v, u, vl, vu, il, iu, abstol, wa2, z, work, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsbevx(V,I,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1370
					}
				}

				//              Do tests 55 and 56 (or +54)
				dsyt22(1, uplo, n, m2, 0, a, wa2, d2, z, v, tau, work, result.Off(ntest-1))

				ntest = ntest + 2

				if uplo == Upper {
					for j = 1; j <= n; j++ {
						for i = max(1, j-kd); i <= j; i++ {
							v.Set(kd+1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				} else {
					for j = 1; j <= n; j++ {
						for i = j; i <= min(n, j+kd); i++ {
							v.Set(1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				}

				*srnamt = "Dsbevx"
				if m3, iinfo, err = golapack.Dsbevx('N', 'I', uplo, n, kd, v, u, vl, vu, il, iu, abstol, wa3, z, work, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsbevx(N,I,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label1370
					}
				}

				//              Do test 57 (or +54)
				temp1 = dsxt1(1, wa2, m2, wa3, m3, abstol, ulp, unfl)
				temp2 = dsxt1(1, wa3, m3, wa2, m2, abstol, ulp, unfl)
				if n > 0 {
					temp3 = math.Max(math.Abs(wa1.Get(0)), math.Abs(wa1.Get(n-1)))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/math.Max(unfl, temp3*ulp))

			label1370:
				;
				ntest = ntest + 1
				if uplo == Upper {
					for j = 1; j <= n; j++ {
						for i = max(1, j-kd); i <= j; i++ {
							v.Set(kd+1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				} else {
					for j = 1; j <= n; j++ {
						for i = j; i <= min(n, j+kd); i++ {
							v.Set(1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				}

				*srnamt = "Dsbevx"
				if m2, iinfo, err = golapack.Dsbevx('V', 'V', uplo, n, kd, v, u, vl, vu, il, iu, abstol, wa2, z, work, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsbevx(V,V,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1460
					}
				}

				//              Do tests 58 and 59 (or +54)
				dsyt22(1, uplo, n, m2, 0, a, wa2, d2, z, v, tau, work, result.Off(ntest-1))

				ntest = ntest + 2

				if uplo == Upper {
					for j = 1; j <= n; j++ {
						for i = max(1, j-kd); i <= j; i++ {
							v.Set(kd+1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				} else {
					for j = 1; j <= n; j++ {
						for i = j; i <= min(n, j+kd); i++ {
							v.Set(1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				}

				*srnamt = "Dsbevx"
				if m3, iinfo, err = golapack.Dsbevx('N', 'V', uplo, n, kd, v, u, vl, vu, il, iu, abstol, wa3, z, work, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsbevx(N,V,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					if err != nil {
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
				temp1 = dsxt1(1, wa2, m2, wa3, m3, abstol, ulp, unfl)
				temp2 = dsxt1(1, wa3, m3, wa2, m2, abstol, ulp, unfl)
				if n > 0 {
					temp3 = math.Max(math.Abs(wa1.Get(0)), math.Abs(wa1.Get(n-1)))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/math.Max(unfl, temp3*ulp))

			label1460:
				;

				//              7)      Call Dsyevd
				golapack.Dlacpy(Full, n, n, a, v)

				ntest = ntest + 1
				*srnamt = "Dsyevd"
				if iinfo, err = golapack.Dsyevd('V', uplo, n, a, d1, work, lwedc, &iwork, liwedc); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsyevd(V,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1480
					}
				}

				//              Do tests 61 and 62 (or +54)
				dsyt21(1, uplo, n, 0, v, d1, d2, a, z, tau, work, result.Off(ntest-1))

				golapack.Dlacpy(Full, n, n, v, a)

				ntest = ntest + 2
				*srnamt = "Dsyevd"
				if iinfo, err = golapack.Dsyevd('N', uplo, n, a, d3, work, lwedc, &iwork, liwedc); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsyevd(N,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
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
					temp1 = math.Max(temp1, math.Max(math.Abs(d1.Get(j-1)), math.Abs(d3.Get(j-1))))
					temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

			label1480:
				;

				//              8)      Call Dspevd.
				golapack.Dlacpy(Full, n, n, v, a)

				//              Load array WORK with the upper or lower triangular
				//              part of the matrix in packed form.
				if uplo == Upper {
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
				*srnamt = "Dspevd"
				if iinfo, err = golapack.Dspevd('V', uplo, n, work, d1, z, work.Off(indx-1), lwedc-indx+1, &iwork, liwedc); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dspevd(V,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1580
					}
				}

				//              Do tests 64 and 65 (or +54)
				dsyt21(1, uplo, n, 0, a, d1, d2, z, v, tau, work, result.Off(ntest-1))

				if uplo == Upper {
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
				*srnamt = "Dspevd"
				if iinfo, err = golapack.Dspevd('N', uplo, n, work, d3, z, work.Off(indx-1), lwedc-indx+1, &iwork, liwedc); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dspevd(N,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
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
					temp1 = math.Max(temp1, math.Max(math.Abs(d1.Get(j-1)), math.Abs(d3.Get(j-1))))
					temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))
			label1580:
				;

				//              9)      Call Dsbevd.
				if jtype <= 7 {
					kd = 1
				} else if jtype >= 8 && jtype <= 15 {
					kd = max(n-1, 0)
				} else {
					kd = ihbw
				}

				//              Load array V with the upper or lower triangular part
				//              of the matrix in band form.
				if uplo == Upper {
					for j = 1; j <= n; j++ {
						for i = max(1, j-kd); i <= j; i++ {
							v.Set(kd+1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				} else {
					for j = 1; j <= n; j++ {
						for i = j; i <= min(n, j+kd); i++ {
							v.Set(1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				}

				ntest = ntest + 1
				*srnamt = "Dsbevd"
				if iinfo, err = golapack.Dsbevd('V', uplo, n, kd, v, d1, z, work, lwedc, &iwork, liwedc); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsbevd(V,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1680
					}
				}

				//              Do tests 67 and 68 (or +54)
				dsyt21(1, uplo, n, 0, a, d1, d2, z, v, tau, work, result.Off(ntest-1))

				if uplo == Upper {
					for j = 1; j <= n; j++ {
						for i = max(1, j-kd); i <= j; i++ {
							v.Set(kd+1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				} else {
					for j = 1; j <= n; j++ {
						for i = j; i <= min(n, j+kd); i++ {
							v.Set(1+i-j-1, j-1, a.Get(i-1, j-1))
						}
					}
				}

				ntest = ntest + 2
				*srnamt = "Dsbevd"
				if iinfo, err = golapack.Dsbevd('N', uplo, n, kd, v, d3, z, work, lwedc, &iwork, liwedc); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsbevd(N,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
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
					temp1 = math.Max(temp1, math.Max(math.Abs(d1.Get(j-1)), math.Abs(d3.Get(j-1))))
					temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

			label1680:
				;

				golapack.Dlacpy(Full, n, n, a, v)
				ntest = ntest + 1
				*srnamt = "Dsyevr"
				if _, iinfo, err = golapack.Dsyevr('V', 'A', uplo, n, a, vl, vu, il, iu, abstol, wa1, z, &iwork, work, lwork, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsyevr(V,A,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1700
					}
				}

				//              Do tests 70 and 71 (or ... )
				golapack.Dlacpy(Full, n, n, v, a)

				dsyt21(1, uplo, n, 0, a, wa1, d2, z, v, tau, work, result.Off(ntest-1))

				ntest = ntest + 2
				*srnamt = "Dsyevr"
				if m2, iinfo, err = golapack.Dsyevr('N', 'A', uplo, n, a, vl, vu, il, iu, abstol, wa2, z, &iwork, work, lwork, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsyevr(N,A,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
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
					temp1 = math.Max(temp1, math.Max(math.Abs(wa1.Get(j-1)), math.Abs(wa2.Get(j-1))))
					temp2 = math.Max(temp2, math.Abs(wa1.Get(j-1)-wa2.Get(j-1)))
				}
				result.Set(ntest-1, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

			label1700:
				;

				ntest = ntest + 1
				golapack.Dlacpy(Full, n, n, v, a)
				*srnamt = "Dsyevr"
				if m2, iinfo, err = golapack.Dsyevr('V', 'I', uplo, n, a, vl, vu, il, iu, abstol, wa2, z, &iwork, work, lwork, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsyevr(V,I,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1710
					}
				}

				//              Do tests 73 and 74 (or +54)
				golapack.Dlacpy(Full, n, n, v, a)

				dsyt22(1, uplo, n, m2, 0, a, wa2, d2, z, v, tau, work, result.Off(ntest-1))

				ntest = ntest + 2
				golapack.Dlacpy(Full, n, n, v, a)
				*srnamt = "Dsyevr"
				if m3, iinfo, err = golapack.Dsyevr('N', 'I', uplo, n, a, vl, vu, il, iu, abstol, wa3, z, &iwork, work, lwork, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsyevr(N,I,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					if err != nil {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label1710
					}
				}

				//              Do test 75 (or +54)
				temp1 = dsxt1(1, wa2, m2, wa3, m3, abstol, ulp, unfl)
				temp2 = dsxt1(1, wa3, m3, wa2, m2, abstol, ulp, unfl)
				result.Set(ntest-1, (temp1+temp2)/math.Max(unfl, ulp*temp3))
			label1710:
				;

				ntest = ntest + 1
				golapack.Dlacpy(Full, n, n, v, a)
				*srnamt = "Dsyevr"
				if m2, iinfo, err = golapack.Dsyevr('V', 'V', uplo, n, a, vl, vu, il, iu, abstol, wa2, z, &iwork, work, lwork, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsyevr(V,V,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label700
					}
				}

				//              Do tests 76 and 77 (or +54)
				golapack.Dlacpy(Full, n, n, v, a)

				dsyt22(1, uplo, n, m2, 0, a, wa2, d2, z, v, tau, work, result.Off(ntest-1))

				ntest = ntest + 2
				golapack.Dlacpy(Full, n, n, v, a)
				*srnamt = "Dsyevr"
				if m3, iinfo, err = golapack.Dsyevr('N', 'V', uplo, n, a, vl, vu, il, iu, abstol, wa3, z, &iwork, work, lwork, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
					t.Fail()
					nerrs++
					nerrs++
					fmt.Printf(" ddrvst: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsyevr(N,V,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
					if err != nil {
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
				temp1 = dsxt1(1, wa2, m2, wa3, m3, abstol, ulp, unfl)
				temp2 = dsxt1(1, wa3, m3, wa2, m2, abstol, ulp, unfl)
				if n > 0 {
					temp3 = math.Max(math.Abs(wa1.Get(0)), math.Abs(wa1.Get(n-1)))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/math.Max(unfl, temp3*ulp))

				golapack.Dlacpy(Full, n, n, v, a)

			}

			//           End of Loop -- Check for RESULT(j) > THRESH
			ntestt = ntestt + ntest

			err = dlafts("Dst", n, n, jtype, ntest, result, ioldsd, thresh, nfails)

		label1730:
		}
	}

	//     Summary
	// alasvm("Dst", nfails, ntestt, nerrs)

	return
}
