package eig

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// zdrvst2stg checks the Hermitian eigenvalue problem drivers.
//
//              Zheevd computes all eigenvalues and, optionally,
//              eigenvectors of a complex Hermitian matrix,
//              using a divide-and-conquer algorithm.
//
//              Zheevx computes selected eigenvalues and, optionally,
//              eigenvectors of a complex Hermitian matrix.
//
//              Zheevr computes selected eigenvalues and, optionally,
//              eigenvectors of a complex Hermitian matrix
//              using the Relatively Robust Representation where it can.
//
//              Zhpevd computes all eigenvalues and, optionally,
//              eigenvectors of a complex Hermitian matrix in packed
//              storage, using a divide-and-conquer algorithm.
//
//              Zhpevx computes selected eigenvalues and, optionally,
//              eigenvectors of a complex Hermitian matrix in packed
//              storage.
//
//              Zhbevd computes all eigenvalues and, optionally,
//              eigenvectors of a complex Hermitian band matrix,
//              using a divide-and-conquer algorithm.
//
//              Zhbevx computes selected eigenvalues and, optionally,
//              eigenvectors of a complex Hermitian band matrix.
//
//              Zheev computes all eigenvalues and, optionally,
//              eigenvectors of a complex Hermitian matrix.
//
//              Zhpev computes all eigenvalues and, optionally,
//              eigenvectors of a complex Hermitian matrix in packed
//              storage.
//
//              Zhbev computes all eigenvalues and, optionally,
//              eigenvectors of a complex Hermitian band matrix.
//
//      When zdrvst2stg is called, a number of matrix "sizes" ("n's") and a
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
func zdrvst2stg(nsizes int, nn []int, ntypes int, dotype []bool, iseed []int, thresh float64, a *mat.CMatrix, d1, d2, d3, wa1, wa2, wa3 *mat.Vector, u, v *mat.CMatrix, tau *mat.CVector, z *mat.CMatrix, work *mat.CVector, lwork int, rwork *mat.Vector, lrwork int, iwork []int, liwork int, result *mat.Vector) (nerrs, ntestt int, err error) {
	var badnn bool
	var uplo mat.MatUplo
	var cone, czero complex128
	var abstol, aninv, anorm, cond, half, one, ovfl, rtovfl, rtunfl, temp1, temp2, temp3, ten, two, ulp, ulpinv, unfl, vl, vu, zero float64
	var i, idiag, ihbw, iinfo, il, imode, indwrk, indx, irow, itemp, itype, iu, iuplo, j, j1, j2, jcol, jsize, jtype, kd, lgn, liwedc, lrwedc, lwedc, m2, m3, maxtyp, mtypes, n, nmats, nmax, ntest int
	idumma := make([]int, 1)
	ioldsd := make([]int, 4)
	iseed2 := make([]int, 4)
	iseed3 := make([]int, 4)
	kmagn := []int{1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3}
	kmode := []int{0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0, 4, 4, 4}
	ktype := []int{1, 2, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 8, 8, 8, 9, 9, 9}

	zero = 0.0
	one = 1.0
	two = 2.0
	ten = 10.0
	half = one / two
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	maxtyp = 18

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
		gltest.Xerbla2("zdrvst2stg", err)
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
	nmats = 0

	for jsize = 1; jsize <= nsizes; jsize++ {
		n = nn[jsize-1]
		if n > 0 {
			lgn = int(math.Log(float64(n)) / math.Log(two))
			if pow(2, lgn) < n {
				lgn = lgn + 1
			}
			if pow(2, lgn) < n {
				lgn = lgn + 1
			}
			lwedc = max(2*n+n*n, 2*n*n)
			lrwedc = 1 + 4*n + 2*n*lgn + 3*pow(n, 2)
			liwedc = 3 + 5*n
		} else {
			lwedc = 2
			lrwedc = 8
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
				goto label1210
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

			golapack.Zlaset(Full, a.Rows, n, czero, czero, a)
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
				matgen.Zlatms(n, n, 'S', &iseed, 'H', rwork, imode, cond, anorm, 0, 0, 'N', a, work)

			} else if itype == 5 {
				//              Hermitian, eigenvalues specified
				matgen.Zlatms(n, n, 'S', &iseed, 'H', rwork, imode, cond, anorm, n, n, 'N', a, work)

			} else if itype == 7 {
				//              Diagonal, random eigenvalues
				matgen.Zlatmr(n, n, 'S', &iseed, 'H', work, 6, one, cone, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, 0, 0, zero, anorm, 'N', a, &iwork)

			} else if itype == 8 {
				//              Hermitian, random eigenvalues
				matgen.Zlatmr(n, n, 'S', &iseed, 'H', work, 6, one, cone, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, n, n, zero, anorm, 'N', a, &iwork)

			} else if itype == 9 {
				//              Hermitian banded, eigenvalues specified
				ihbw = int(float64(n-1) * matgen.Dlarnd(1, &iseed3))
				err = matgen.Zlatms(n, n, 'S', &iseed, 'H', rwork, imode, cond, anorm, ihbw, ihbw, 'Z', u, work)

				//              Store as dense matrix for most routines.
				golapack.Zlaset(Full, a.Rows, n, czero, czero, a)
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

			if iinfo != 0 || err != nil {
				fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", "Generator", iinfo, n, jtype, ioldsd)
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
				il = 1 + int(float64(n-1)*matgen.Dlarnd(1, &iseed2))
				iu = 1 + int(float64(n-1)*matgen.Dlarnd(1, &iseed2))
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
					uplo = Lower
				} else {
					uplo = Upper
				}

				//              Call Zheevd and CHEEVX.
				golapack.Zlacpy(Full, n, n, a, v)
				//
				ntest = ntest + 1
				if iinfo, err = golapack.Zheevd('V', uplo, n, a, d1, work, lwedc, rwork, lrwedc, &iwork, liwedc); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zheevd(V,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label130
					}
				}

				//              Do tests 1 and 2.
				zhet21(1, uplo, n, 0, v, d1, d2, a, z, tau, work, rwork, result.Off(ntest-1))

				golapack.Zlacpy(Full, n, n, v, a)

				ntest = ntest + 2
				if iinfo, err = golapack.Zheevd2stage('N', uplo, n, a, d3, work, lwork, rwork, lrwedc, &iwork, liwedc); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zheevd2stage(N,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
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
					temp1 = math.Max(temp1, math.Max(d1.GetMag(j-1), d3.GetMag(j-1)))
					temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

			label130:
				;
				golapack.Zlacpy(Full, n, n, v, a)

				ntest = ntest + 1

				if n > 0 {
					temp3 = math.Max(d1.GetMag(0), d1.GetMag(n-1))
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

				if _, iinfo, err = golapack.Zheevx('V', 'A', uplo, n, a, vl, vu, il, iu, abstol, wa1, z, work, lwork, rwork, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zheevx(V,A,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label150
					}
				}

				//              Do tests 4 and 5.
				golapack.Zlacpy(Full, n, n, v, a)

				zhet21(1, uplo, n, 0, a, wa1, d2, z, v, tau, work, rwork, result.Off(ntest-1))

				ntest = ntest + 2
				if m2, iinfo, err = golapack.Zheevx2stage('N', 'A', uplo, n, a, vl, vu, il, iu, abstol, wa2, z, work, lwork, rwork, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zheevx2stage(N,A,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
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
					temp1 = math.Max(temp1, math.Max(wa1.GetMag(j-1), wa2.GetMag(j-1)))
					temp2 = math.Max(temp2, math.Abs(wa1.Get(j-1)-wa2.Get(j-1)))
				}
				result.Set(ntest-1, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

			label150:
				;
				golapack.Zlacpy(Full, n, n, v, a)

				ntest = ntest + 1

				if m2, iinfo, err = golapack.Zheevx('V', 'I', uplo, n, a, vl, vu, il, iu, abstol, wa2, z, work, lwork, rwork, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zheevx(V,I,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label160
					}
				}

				//              Do tests 7 and 8.
				golapack.Zlacpy(Full, n, n, v, a)

				zhet22(1, uplo, n, m2, 0, a, wa2, d2, z, v, tau, work, rwork, result.Off(ntest-1))

				ntest = ntest + 2

				if m3, iinfo, err = golapack.Zheevx2stage('N', 'I', uplo, n, a, vl, vu, il, iu, abstol, wa3, z, work, lwork, rwork, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zheevx2stage(N,I,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label160
					}
				}

				//              Do test 9.
				temp1 = dsxt1(1, wa2, m2, wa3, m3, abstol, ulp, unfl)
				temp2 = dsxt1(1, wa3, m3, wa2, m2, abstol, ulp, unfl)
				if n > 0 {
					temp3 = math.Max(wa1.GetMag(0), wa1.GetMag(n-1))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/math.Max(unfl, temp3*ulp))

			label160:
				;
				golapack.Zlacpy(Full, n, n, v, a)

				ntest = ntest + 1

				if m2, iinfo, err = golapack.Zheevx('V', 'V', uplo, n, a, vl, vu, il, iu, abstol, wa2, z, work, lwork, rwork, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zheevx(V,V,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label170
					}
				}

				//              Do tests 10 and 11.
				golapack.Zlacpy(Full, n, n, v, a)

				zhet22(1, uplo, n, m2, 0, a, wa2, d2, z, v, tau, work, rwork, result.Off(ntest-1))

				ntest = ntest + 2

				if m3, iinfo, err = golapack.Zheevx2stage('N', 'V', uplo, n, a, vl, vu, il, iu, abstol, wa3, z, work, lwork, rwork, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zheevx2stage(N,V,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
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
				temp1 = dsxt1(1, wa2, m2, wa3, m3, abstol, ulp, unfl)
				temp2 = dsxt1(1, wa3, m3, wa2, m2, abstol, ulp, unfl)
				if n > 0 {
					temp3 = math.Max(wa1.GetMag(0), wa1.GetMag(n-1))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/math.Max(unfl, temp3*ulp))

			label170:
				;

				//              Call Zhpevd and CHPEVX.
				golapack.Zlacpy(Full, n, n, v, a)

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
				if iinfo, err = golapack.Zhpevd('V', uplo, n, work, d1, z, work.Off(indwrk-1), lwedc, rwork, lrwedc, &iwork, liwedc); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zhpevd(V,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label270
					}
				}

				//              Do tests 13 and 14.
				zhet21(1, uplo, n, 0, a, d1, d2, z, v, tau, work, rwork, result.Off(ntest-1))

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
				if iinfo, err = golapack.Zhpevd('N', uplo, n, work, d3, z, work.Off(indwrk-1), lwedc, rwork, lrwedc, &iwork, liwedc); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zhpevd(N,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
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
					temp1 = math.Max(temp1, math.Max(d1.GetMag(j-1), d3.GetMag(j-1)))
					temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

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
					temp3 = math.Max(d1.GetMag(0), d1.GetMag(n-1))
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

				if _, iinfo, err = golapack.Zhpevx('V', 'A', uplo, n, work, vl, vu, il, iu, abstol, wa1, z, v.Off(0, 0).CVector(), rwork, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zhpevx(V,A,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label370
					}
				}

				//              Do tests 16 and 17.
				zhet21(1, uplo, n, 0, a, wa1, d2, z, v, tau, work, rwork, result.Off(ntest-1))

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

				if m2, iinfo, err = golapack.Zhpevx('N', 'A', uplo, n, work, vl, vu, il, iu, abstol, wa2, z, v.Off(0, 0).CVector(), rwork, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zhpevx(N,A,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
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
					temp1 = math.Max(temp1, math.Max(wa1.GetMag(j-1), wa2.GetMag(j-1)))
					temp2 = math.Max(temp2, math.Abs(wa1.Get(j-1)-wa2.Get(j-1)))
				}
				result.Set(ntest-1, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

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

				if m2, iinfo, err = golapack.Zhpevx('V', 'I', uplo, n, work, vl, vu, il, iu, abstol, wa2, z, v.Off(0, 0).CVector(), rwork, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zhpevx(V,I,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label460
					}
				}

				//              Do tests 19 and 20.
				zhet22(1, uplo, n, m2, 0, a, wa2, d2, z, v, tau, work, rwork, result.Off(ntest-1))

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

				if m3, iinfo, err = golapack.Zhpevx('N', 'I', uplo, n, work, vl, vu, il, iu, abstol, wa3, z, v.Off(0, 0).CVector(), rwork, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zhpevx(N,I,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label460
					}
				}

				//              Do test 21.
				temp1 = dsxt1(1, wa2, m2, wa3, m3, abstol, ulp, unfl)
				temp2 = dsxt1(1, wa3, m3, wa2, m2, abstol, ulp, unfl)
				if n > 0 {
					temp3 = math.Max(wa1.GetMag(0), wa1.GetMag(n-1))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/math.Max(unfl, temp3*ulp))

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

				if m2, iinfo, err = golapack.Zhpevx('V', 'V', uplo, n, work, vl, vu, il, iu, abstol, wa2, z, v.Off(0, 0).CVector(), rwork, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zhpevx(V,V,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label550
					}
				}

				//              Do tests 22 and 23.
				zhet22(1, uplo, n, m2, 0, a, wa2, d2, z, v, tau, work, rwork, result.Off(ntest-1))

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

				if m3, iinfo, err = golapack.Zhpevx('N', 'V', uplo, n, work, vl, vu, il, iu, abstol, wa3, z, v.Off(0, 0).CVector(), rwork, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zhpevx(N,V,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
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
				temp1 = dsxt1(1, wa2, m2, wa3, m3, abstol, ulp, unfl)
				temp2 = dsxt1(1, wa3, m3, wa2, m2, abstol, ulp, unfl)
				if n > 0 {
					temp3 = math.Max(wa1.GetMag(0), wa1.GetMag(n-1))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/math.Max(unfl, temp3*ulp))

			label550:
				;

				//              Call Zhbevd and CHBEVX.
				if jtype <= 7 {
					kd = 0
				} else if jtype >= 8 && jtype <= 15 {
					kd = max(n-1, 0)
				} else {
					kd = ihbw
				}

				//              Load array V with the upper or lower triangular part
				//              of the matrix in band form.
				if iuplo == 1 {
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
				if iinfo, err = golapack.Zhbevd('V', uplo, n, kd, v, d1, z, work, lwedc, rwork, lrwedc, &iwork, liwedc); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, KD=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zhbevd(V,"), uplo.Byte(), ')'), iinfo, n, kd, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label650
					}
				}

				//              Do tests 25 and 26.
				zhet21(1, uplo, n, 0, a, d1, d2, z, v, tau, work, rwork, result.Off(ntest-1))

				if iuplo == 1 {
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
				if iinfo, err = golapack.Zhbevd2stage('N', uplo, n, kd, v, d3, z, work, lwork, rwork, lrwedc, &iwork, liwedc); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, KD=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zhbevd2stage(N,"), uplo.Byte(), ')'), iinfo, n, kd, jtype, ioldsd)
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
					temp1 = math.Max(temp1, math.Max(d1.GetMag(j-1), d3.GetMag(j-1)))
					temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

				//              Load array V with the upper or lower triangular part
				//              of the matrix in band form.
			label650:
				;
				if iuplo == 1 {
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
				if _, iinfo, err = golapack.Zhbevx('V', 'A', uplo, n, kd, v, u, vl, vu, il, iu, abstol, wa1, z, work, rwork, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zhbevx(V,A,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label750
					}
				}

				//              Do tests 28 and 29.
				zhet21(1, uplo, n, 0, a, wa1, d2, z, v, tau, work, rwork, result.Off(ntest-1))

				ntest = ntest + 2

				if iuplo == 1 {
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

				if m2, iinfo, err = golapack.Zhbevx2stage('N', 'A', uplo, n, kd, v, u, vl, vu, il, iu, abstol, wa2, z, work, lwork, rwork, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, KD=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zhbevx2stage(N,A,"), uplo.Byte(), ')'), iinfo, n, kd, jtype, ioldsd)
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
					temp1 = math.Max(temp1, math.Max(wa1.GetMag(j-1), wa2.GetMag(j-1)))
					temp2 = math.Max(temp2, math.Abs(wa1.Get(j-1)-wa2.Get(j-1)))
				}
				result.Set(ntest-1, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

				//              Load array V with the upper or lower triangular part
				//              of the matrix in band form.
			label750:
				;
				ntest = ntest + 1
				if iuplo == 1 {
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

				if m2, iinfo, err = golapack.Zhbevx('V', 'I', uplo, n, kd, v, u, vl, vu, il, iu, abstol, wa2, z, work, rwork, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, KD=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zhbevx(V,I,"), uplo.Byte(), ')'), iinfo, n, kd, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label840
					}
				}

				//              Do tests 31 and 32.
				zhet22(1, uplo, n, m2, 0, a, wa2, d2, z, v, tau, work, rwork, result.Off(ntest-1))

				ntest = ntest + 2

				if iuplo == 1 {
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
				if m3, iinfo, err = golapack.Zhbevx2stage('N', 'I', uplo, n, kd, v, u, vl, vu, il, iu, abstol, wa3, z, work, lwork, rwork, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, KD=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zhbevx2stage(N,I,"), uplo.Byte(), ')'), iinfo, n, kd, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label840
					}
				}

				//              Do test 33.
				temp1 = dsxt1(1, wa2, m2, wa3, m3, abstol, ulp, unfl)
				temp2 = dsxt1(1, wa3, m3, wa2, m2, abstol, ulp, unfl)
				if n > 0 {
					temp3 = math.Max(wa1.GetMag(0), wa1.GetMag(n-1))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/math.Max(unfl, temp3*ulp))

				//              Load array V with the upper or lower triangular part
				//              of the matrix in band form.
			label840:
				;
				ntest = ntest + 1
				if iuplo == 1 {
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
				if m2, iinfo, err = golapack.Zhbevx('V', 'V', uplo, n, kd, v, u, vl, vu, il, iu, abstol, wa2, z, work, rwork, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, KD=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zhbevx(V,V,"), uplo.Byte(), ')'), iinfo, n, kd, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label930
					}
				}

				//              Do tests 34 and 35.
				zhet22(1, uplo, n, m2, 0, a, wa2, d2, z, v, tau, work, rwork, result.Off(ntest-1))

				ntest = ntest + 2

				if iuplo == 1 {
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
				if m3, iinfo, err = golapack.Zhbevx2stage('N', 'V', uplo, n, kd, v, u, vl, vu, il, iu, abstol, wa3, z, work, lwork, rwork, &iwork, toSlice(&iwork, 5*n)); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, KD=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zhbevx2stage(N,V,"), uplo.Byte(), ')'), iinfo, n, kd, jtype, ioldsd)
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
				temp1 = dsxt1(1, wa2, m2, wa3, m3, abstol, ulp, unfl)
				temp2 = dsxt1(1, wa3, m3, wa2, m2, abstol, ulp, unfl)
				if n > 0 {
					temp3 = math.Max(wa1.GetMag(0), wa1.GetMag(n-1))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/math.Max(unfl, temp3*ulp))

			label930:
				;

				//              Call Zheev
				golapack.Zlacpy(Full, n, n, a, v)

				ntest = ntest + 1
				if iinfo, err = golapack.Zheev('V', uplo, n, a, d1, work, lwork, rwork); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zheev(V,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label950
					}
				}

				//              Do tests 37 and 38
				zhet21(1, uplo, n, 0, v, d1, d2, a, z, tau, work, rwork, result.Off(ntest-1))

				golapack.Zlacpy(Full, n, n, v, a)

				ntest = ntest + 2
				if iinfo, err = golapack.Zheev2stage('N', uplo, n, a, d3, work, lwork, rwork); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zheev2stage(N,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
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
					temp1 = math.Max(temp1, math.Max(d1.GetMag(j-1), d3.GetMag(j-1)))
					temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

			label950:
				;

				golapack.Zlacpy(Full, n, n, v, a)

				//              Call Zhpev
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
				if iinfo, err = golapack.Zhpev('V', uplo, n, work, d1, z, work.Off(indwrk-1), rwork); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zhpev(V,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1050
					}
				}

				//              Do tests 40 and 41.
				zhet21(1, uplo, n, 0, a, d1, d2, z, v, tau, work, rwork, result.Off(ntest-1))

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
				if iinfo, err = golapack.Zhpev('N', uplo, n, work, d3, z, work.Off(indwrk-1), rwork); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zhpev(N,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
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
					temp1 = math.Max(temp1, math.Max(d1.GetMag(j-1), d3.GetMag(j-1)))
					temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

			label1050:
				;

				//              Call Zhbev
				if jtype <= 7 {
					kd = 0
				} else if jtype >= 8 && jtype <= 15 {
					kd = max(n-1, 0)
				} else {
					kd = ihbw
				}

				//              Load array V with the upper or lower triangular part
				//              of the matrix in band form.
				if iuplo == 1 {
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
				//
				ntest = ntest + 1
				if iinfo, err = golapack.Zhbev('V', uplo, n, kd, v, d1, z, work, rwork); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, KD=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zhbev(V,"), uplo.Byte(), ')'), iinfo, n, kd, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1140
					}
				}

				//              Do tests 43 and 44.
				zhet21(1, uplo, n, 0, a, d1, d2, z, v, tau, work, rwork, result.Off(ntest-1))

				if iuplo == 1 {
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
				if iinfo, err = golapack.Zhbev2stage('N', uplo, n, kd, v, d3, z, work, lwork, rwork); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, KD=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zhbev2stage(N,"), uplo.Byte(), ')'), iinfo, n, kd, jtype, ioldsd)
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
					temp1 = math.Max(temp1, math.Max(d1.GetMag(j-1), d3.GetMag(j-1)))
					temp2 = math.Max(temp2, math.Abs(d1.Get(j-1)-d3.Get(j-1)))
				}
				result.Set(ntest-1, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

				golapack.Zlacpy(Full, n, n, a, v)
				ntest = ntest + 1
				if _, iinfo, err = golapack.Zheevr('V', 'A', uplo, n, a, vl, vu, il, iu, abstol, wa1, z, &iwork, work, lwork, rwork, lrwork, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zheevr(V,A,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1170
					}
				}

				//              Do tests 45 and 46 (or ... )
				golapack.Zlacpy(Full, n, n, v, a)

				zhet21(1, uplo, n, 0, a, wa1, d2, z, v, tau, work, rwork, result.Off(ntest-1))

				ntest = ntest + 2
				if m2, iinfo, err = golapack.Zheevr2stage('N', 'A', uplo, n, a, vl, vu, il, iu, abstol, wa2, z, &iwork, work, lwork, rwork, lrwork, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zheevr2stage(N,A,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
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
					temp1 = math.Max(temp1, math.Max(wa1.GetMag(j-1), wa2.GetMag(j-1)))
					temp2 = math.Max(temp2, math.Abs(wa1.Get(j-1)-wa2.Get(j-1)))
				}
				result.Set(ntest-1, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

			label1170:
				;

				ntest = ntest + 1
				golapack.Zlacpy(Full, n, n, v, a)
				if m2, iinfo, err = golapack.Zheevr('V', 'I', uplo, n, a, vl, vu, il, iu, abstol, wa2, z, &iwork, work, lwork, rwork, lrwork, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zheevr(V,I,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1180
					}
				}

				//              Do tests 48 and 49 (or +??)
				golapack.Zlacpy(Full, n, n, v, a)

				zhet22(1, uplo, n, m2, 0, a, wa2, d2, z, v, tau, work, rwork, result.Off(ntest-1))

				ntest = ntest + 2
				golapack.Zlacpy(Full, n, n, v, a)
				if m3, iinfo, err = golapack.Zheevr2stage('N', 'I', uplo, n, a, vl, vu, il, iu, abstol, wa3, z, &iwork, work, lwork, rwork, lrwork, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zheevr2stage(N,I,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						goto label1180
					}
				}

				//              Do test 50 (or +??)
				temp1 = dsxt1(1, wa2, m2, wa3, m3, abstol, ulp, unfl)
				temp2 = dsxt1(1, wa3, m3, wa2, m2, abstol, ulp, unfl)
				result.Set(ntest-1, (temp1+temp2)/math.Max(unfl, ulp*temp3))
			label1180:
				;

				ntest = ntest + 1
				golapack.Zlacpy(Full, n, n, v, a)
				if m2, iinfo, err = golapack.Zheevr('V', 'V', uplo, n, a, vl, vu, il, iu, abstol, wa2, z, &iwork, work, lwork, rwork, lrwork, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zheevr(V,V,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
					if iinfo < 0 {
						return
					} else {
						result.Set(ntest-1, ulpinv)
						result.Set(ntest, ulpinv)
						result.Set(ntest+2-1, ulpinv)
						goto label1190
					}
				}

				//              Do tests 51 and 52 (or +??)
				golapack.Zlacpy(Full, n, n, v, a)

				zhet22(1, uplo, n, m2, 0, a, wa2, d2, z, v, tau, work, rwork, result.Off(ntest-1))

				ntest = ntest + 2
				golapack.Zlacpy(Full, n, n, v, a)
				if m3, iinfo, err = golapack.Zheevr2stage('N', 'V', uplo, n, a, vl, vu, il, iu, abstol, wa3, z, &iwork, work, lwork, rwork, lrwork, toSlice(&iwork, 2*n), liwork-2*n); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvst2stg: %s returned info=%6d\n         n=%6d, jtype=%6d, iseed=%5d\n", append([]byte("Zheevr2stage(N,V,"), uplo.Byte(), ')'), iinfo, n, jtype, ioldsd)
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
				temp1 = dsxt1(1, wa2, m2, wa3, m3, abstol, ulp, unfl)
				temp2 = dsxt1(1, wa3, m3, wa2, m2, abstol, ulp, unfl)
				if n > 0 {
					temp3 = math.Max(wa1.GetMag(0), wa1.GetMag(n-1))
				} else {
					temp3 = zero
				}
				result.Set(ntest-1, (temp1+temp2)/math.Max(unfl, temp3*ulp))

				golapack.Zlacpy(Full, n, n, v, a)

				//              Load array V with the upper or lower triangular part
				//              of the matrix in band form.
			label1190:
			}

			//           End of Loop -- Check for RESULT(j) > THRESH
			ntestt = ntestt + ntest
			err = dlafts("Zst", n, n, jtype, ntest, result, ioldsd, thresh, nerrs)

		label1210:
		}
	}

	//     Summary
	// alasvm("Zst", nerrs, ntestt, 0)

	return
}
