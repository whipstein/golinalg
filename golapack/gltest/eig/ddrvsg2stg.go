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

// ddrvsg2stg checks the real symmetric generalized eigenproblem
//      drivers.
//
//              Dsygv computes all eigenvalues and, optionally,
//              eigenvectors of a real symmetric-definite generalized
//              eigenproblem.
//
//              Dsygvd computes all eigenvalues and, optionally,
//              eigenvectors of a real symmetric-definite generalized
//              eigenproblem using a divide and conquer algorithm.
//
//              Dsygvx computes selected eigenvalues and, optionally,
//              eigenvectors of a real symmetric-definite generalized
//              eigenproblem.
//
//              Dspgv computes all eigenvalues and, optionally,
//              eigenvectors of a real symmetric-definite generalized
//              eigenproblem in packed storage.
//
//              Dspgvd computes all eigenvalues and, optionally,
//              eigenvectors of a real symmetric-definite generalized
//              eigenproblem in packed storage using a divide and
//              conquer algorithm.
//
//              Dspgvx computes selected eigenvalues and, optionally,
//              eigenvectors of a real symmetric-definite generalized
//              eigenproblem in packed storage.
//
//              Dsbgv computes all eigenvalues and, optionally,
//              eigenvectors of a real symmetric-definite banded
//              generalized eigenproblem.
//
//              Dsbgvd computes all eigenvalues and, optionally,
//              eigenvectors of a real symmetric-definite banded
//              generalized eigenproblem using a divide and conquer
//              algorithm.
//
//              Dsbgvx computes selected eigenvalues and, optionally,
//              eigenvectors of a real symmetric-definite banded
//              generalized eigenproblem.
//
//      When ddrvsg2stg is called, a number of matrix "sizes" ("n's") and a
//      number of matrix "types" are specified.  For each size ("n")
//      and each type of matrix, one matrix A of the given type will be
//      generated; a random well-conditioned matrix B is also generated
//      and the pair (A,B) is used to test the drivers.
//
//      For each pair (A,B), the following tests are performed:
//
//      (1) Dsygv with ITYPE = 1 and UPLO ='U':
//
//              | A Z - B Z D | / ( |A| |Z| n ulp )
//              | D - D2 | / ( |D| ulp )   where D is computed by
//                                         Dsygv and  D2 is computed by
//                                         Dsygv2stage. This test is
//                                         only performed for Dsygv
//
//      (2) as (1) but calling Dspgv
//      (3) as (1) but calling Dsbgv
//      (4) as (1) but with UPLO = 'L'
//      (5) as (4) but calling Dspgv
//      (6) as (4) but calling Dsbgv
//
//      (7) Dsygv with ITYPE = 2 and UPLO ='U':
//
//              | A B Z - Z D | / ( |A| |Z| n ulp )
//
//      (8) as (7) but calling Dspgv
//      (9) as (7) but with UPLO = 'L'
//      (10) as (9) but calling Dspgv
//
//      (11) Dsygv with ITYPE = 3 and UPLO ='U':
//
//              | B A Z - Z D | / ( |A| |Z| n ulp )
//
//      (12) as (11) but calling Dspgv
//      (13) as (11) but with UPLO = 'L'
//      (14) as (13) but calling Dspgv
//
//      Dsygvd, Dspgvd and Dsbgvd performed the same 14 tests.
//
//      Dsygvx, Dspgvx and Dsbgvx performed the above 14 tests with
//      the parameter RANGE = 'A', 'N' and 'I', respectively.
//
//      The "sizes" are specified by an array NN(1:NSIZES); the value
//      of each element NN(j) specifies one size.
//      The "types" are specified by a logical array DOTYPE( 1:NTYPES );
//      if DOTYPE(j) is .TRUE., then matrix type "j" will be generated.
//      This type is used for the matrix A which has half-bandwidth KA.
//      B is generated as a well-conditioned positive definite matrix
//      with half-bandwidth KB (<= KA).
//      Currently, the list of possible types for A is:
//
//      (1)  The zero matrix.
//      (2)  The identity matrix.
//
//      (3)  A diagonal matrix with evenly spaced entries
//           1, ..., ULP  and random signs.
//           (ULP = (first number larger than 1) - 1 )
//      (4)  A diagonal matrix with geometrically spaced entries
//           1, ..., ULP  and random signs.
//      (5)  A diagonal matrix with "clustered" entries
//           1, ULP, ..., ULP and random signs.
//
//      (6)  Same as (4), but multiplied by SQRT( overflow threshold )
//      (7)  Same as (4), but multiplied by SQRT( underflow threshold )
//
//      (8)  A matrix of the form  U* D U, where U is orthogonal and
//           D has evenly spaced entries 1, ..., ULP with random signs
//           on the diagonal.
//
//      (9)  A matrix of the form  U* D U, where U is orthogonal and
//           D has geometrically spaced entries 1, ..., ULP with random
//           signs on the diagonal.
//
//      (10) A matrix of the form  U* D U, where U is orthogonal and
//           D has "clustered" entries 1, ULP,..., ULP with random
//           signs on the diagonal.
//
//      (11) Same as (8), but multiplied by SQRT( overflow threshold )
//      (12) Same as (8), but multiplied by SQRT( underflow threshold )
//
//      (13) symmetric matrix with random entries chosen from (-1,1).
//      (14) Same as (13), but multiplied by SQRT( overflow threshold )
//      (15) Same as (13), but multiplied by SQRT( underflow threshold)
//
//      (16) Same as (8), but with KA = 1 and KB = 1
//      (17) Same as (8), but with KA = 2 and KB = 1
//      (18) Same as (8), but with KA = 2 and KB = 2
//      (19) Same as (8), but with KA = 3 and KB = 1
//      (20) Same as (8), but with KA = 3 and KB = 2
//      (21) Same as (8), but with KA = 3 and KB = 3
func ddrvsg2stg(nsizes int, nn []int, ntypes int, dotype []bool, iseed []int, thresh float64, nounit int, a, b *mat.Matrix, d, d2 *mat.Vector, z, ab, bb *mat.Matrix, ap, bp, work *mat.Vector, nwork int, iwork []int, liwork int, result *mat.Vector, t *testing.T) (err error) {
	var badnn bool
	var uplo mat.MatUplo
	var abstol, aninv, anorm, cond, one, ovfl, rtovfl, rtunfl, temp1, temp2, ten, ulp, ulpinv, unfl, vl, vu, zero float64
	var i, ibtype, ibuplo, iinfo, ij, il, imode, itemp, itype, iu, j, jcol, jsize, jtype, ka, ka9, kb, kb9, m, maxtyp, mtypes, n, nerrs, nmats, nmax, ntest, ntestt int

	idumma := make([]int, 1)
	ioldsd := make([]int, 4)
	iseed2 := make([]int, 4)
	kmagn := make([]int, 21)
	kmode := make([]int, 21)
	ktype := make([]int, 21)

	zero = 0.0
	one = 1.0
	ten = 10.0
	maxtyp = 21

	ktype[0], ktype[1], ktype[2], ktype[3], ktype[4], ktype[5], ktype[6], ktype[7], ktype[8], ktype[9], ktype[10], ktype[11], ktype[12], ktype[13], ktype[14], ktype[15], ktype[16], ktype[17], ktype[18], ktype[19], ktype[20] = 1, 2, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 8, 8, 8, 9, 9, 9, 9, 9, 9
	kmagn[0], kmagn[1], kmagn[2], kmagn[3], kmagn[4], kmagn[5], kmagn[6], kmagn[7], kmagn[8], kmagn[9], kmagn[10], kmagn[11], kmagn[12], kmagn[13], kmagn[14], kmagn[15], kmagn[16], kmagn[17], kmagn[18], kmagn[19], kmagn[20] = 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 1, 1
	kmode[0], kmode[1], kmode[2], kmode[3], kmode[4], kmode[5], kmode[6], kmode[7], kmode[8], kmode[9], kmode[10], kmode[11], kmode[12], kmode[13], kmode[14], kmode[15], kmode[16], kmode[17], kmode[18], kmode[19], kmode[20] = 0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0, 4, 4, 4, 4, 4, 4

	//     1)      Check for errors
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
	} else if a.Rows <= 1 || a.Rows < nmax {
		err = fmt.Errorf("a.Rows <= 1 || a.Rows < nmax: a.Rows=%v, nmax=%v", a.Rows, nmax)
	} else if z.Rows <= 1 || z.Rows < nmax {
		err = fmt.Errorf("z.Rows <= 1 || z.Rows < nmax: z.Rows=%v, nmax=%v", z.Rows, nmax)
	} else if 2*pow(max(nmax, 3), 2) > nwork {
		err = fmt.Errorf("2*pow(max(nmax, 3), 2) > nwork: nmax=%v, nwork=%v", nmax, nwork)
	} else if 2*pow(max(nmax, 3), 2) > liwork {
		err = fmt.Errorf("2*pow(max(nmax, 3), 2) > liwork: nmax=%v, liwork=%v", nmax, liwork)
	}

	if err != nil {
		gltest.Xerbla2("ddrvsg2stg", err)
		return
	}

	//     Quick return if possible
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

	for i = 1; i <= 4; i++ {
		iseed2[i-1] = iseed[i-1]
	}

	//     Loop over sizes, types
	nerrs = 0
	nmats = 0

	for jsize = 1; jsize <= nsizes; jsize++ {
		n = nn[jsize-1]
		aninv = one / float64(max(1, n))

		if nsizes != 1 {
			mtypes = min(maxtyp, ntypes)
		} else {
			mtypes = min(maxtyp+1, ntypes)
		}

		ka9 = 0
		kb9 = 0
		for jtype = 1; jtype <= mtypes; jtype++ {
			if !dotype[jtype-1] {
				goto label640
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
			//           =4         arithmetic   diagonal, w/ eigenvalues
			//           =5         random log   hermitian, w/ eigenvalues
			//           =6         random       (none)
			//           =7                      random diagonal
			//           =8                      random hermitian
			//           =9                      banded, w/ eigenvalues
			if mtypes > maxtyp {
				goto label90
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

			iinfo = 0
			cond = ulpinv

			//           Special Matrices -- Identity & Jordan block
			if itype == 1 {
				//              Zero
				ka = 0
				kb = 0
				golapack.Dlaset(Full, a.Rows, n, zero, zero, a)

			} else if itype == 2 {
				//              Identity
				ka = 0
				kb = 0
				golapack.Dlaset(Full, a.Rows, n, zero, zero, a)
				for jcol = 1; jcol <= n; jcol++ {
					a.Set(jcol-1, jcol-1, anorm)
				}

			} else if itype == 4 {
				//              Diagonal Matrix, [Eigen]values Specified
				ka = 0
				kb = 0
				iinfo, err = matgen.Dlatms(n, n, 'S', &iseed, 'S', work, imode, cond, anorm, 0, 0, 'N', a, work.Off(n))

			} else if itype == 5 {
				//              symmetric, eigenvalues specified
				ka = max(0, n-1)
				kb = ka
				iinfo, err = matgen.Dlatms(n, n, 'S', &iseed, 'S', work, imode, cond, anorm, n, n, 'N', a, work.Off(n))

			} else if itype == 7 {
				//              Diagonal, random eigenvalues
				ka = 0
				kb = 0
				iinfo, err = matgen.Dlatmr(n, n, 'S', &iseed, 'S', work, 6, one, one, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, 0, 0, zero, anorm, 'N', a, &iwork)

			} else if itype == 8 {
				//              symmetric, random eigenvalues
				ka = max(0, n-1)
				kb = ka
				iinfo, err = matgen.Dlatmr(n, n, 'S', &iseed, 'H', work, 6, one, one, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, n, n, zero, anorm, 'N', a, &iwork)

			} else if itype == 9 {
				//              symmetric banded, eigenvalues specified
				//
				//              The following values are used for the half-bandwidths:
				//
				//                ka = 1   kb = 1
				//                ka = 2   kb = 1
				//                ka = 2   kb = 2
				//                ka = 3   kb = 1
				//                ka = 3   kb = 2
				//                ka = 3   kb = 3
				kb9 = kb9 + 1
				if kb9 > ka9 {
					ka9 = ka9 + 1
					kb9 = 1
				}
				ka = max(0, min(n-1, ka9))
				kb = max(0, min(n-1, kb9))
				iinfo, err = matgen.Dlatms(n, n, 'S', &iseed, 'S', work, imode, cond, anorm, ka, ka, 'N', a, work.Off(n))

			} else {

				iinfo = 1
			}

			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ddrvsg2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Generator", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				return
			}

		label90:
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

			//           3) Call Dsygv, Dspgv, Dsbgv, SSYGVD, SSPGVD, SSBGVD,
			//              Dsygvx, Dspgvx, and Dsbgvx, do tests.
			//
			//           loop over the three generalized problems
			//                 IBTYPE = 1: A*x = (lambda)*B*x
			//                 IBTYPE = 2: A*B*x = (lambda)*x
			//                 IBTYPE = 3: B*A*x = (lambda)*x
			for ibtype = 1; ibtype <= 3; ibtype++ {
				//              loop over the setting UPLO
				for ibuplo = 1; ibuplo <= 2; ibuplo++ {
					if ibuplo == 1 {
						uplo = Upper
					}
					if ibuplo == 2 {
						uplo = Lower
					}

					//                 Generate random well-conditioned positive definite
					//                 matrix B, of bandwidth not greater than that of A.
					matgen.Dlatms(n, n, 'U', &iseed, 'P', work, 5, ten, one, kb, kb, uplo.Byte(), b, work.Off(n))

					//                 Test Dsygv
					ntest = ntest + 1

					golapack.Dlacpy(Full, n, n, a, z)
					golapack.Dlacpy(uplo, n, n, b, bb)

					if iinfo, err = golapack.Dsygv(ibtype, 'V', uplo, n, z, bb, d, work, nwork); err != nil || iinfo != 0 {
						t.Fail()
						fmt.Printf(" ddrvsg2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsygv(V,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
						if iinfo < 0 {
							return
						} else {
							result.Set(ntest-1, ulpinv)
							goto label100
						}
					}

					//                 Do Test
					dsgt01(ibtype, uplo, n, n, a, b, z, d, work, result.Off(ntest-1))

					//                 Test Dsygv2stage
					ntest = ntest + 1

					golapack.Dlacpy(Full, n, n, a, z)
					golapack.Dlacpy(uplo, n, n, b, bb)

					if iinfo, err = golapack.Dsygv2stage(ibtype, 'N', uplo, n, z, bb, d2, work, nwork); err != nil || iinfo != 0 {
						t.Fail()
						fmt.Printf(" ddrvsg2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsygv2stage(V,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
						if iinfo < 0 {
							return
						} else {
							result.Set(ntest-1, ulpinv)
							goto label100
						}
					}

					//                 Do Test
					//
					//C                  CALL DSGT01( IBTYPE, UPLO, N, N, A, LDA, B, LDB, Z,
					//C     $                         LDZ, D, WORK, RESULT( NTEST ) )
					//
					//                 Do Tests | D1 - D2 | / ( |D1| ulp )
					//                 D1 computed using the standard 1-stage reduction as reference
					//                 D2 computed using the 2-stage reduction
					temp1 = zero
					temp2 = zero
					for j = 1; j <= n; j++ {
						temp1 = math.Max(temp1, math.Max(math.Abs(d.Get(j-1)), math.Abs(d2.Get(j-1))))
						temp2 = math.Max(temp2, math.Abs(d.Get(j-1)-d2.Get(j-1)))
					}

					result.Set(ntest-1, temp2/math.Max(unfl, ulp*math.Max(temp1, temp2)))

					//                 Test Dsygvd
					ntest = ntest + 1

					golapack.Dlacpy(Full, n, n, a, z)
					golapack.Dlacpy(uplo, n, n, b, bb)

					if iinfo, err = golapack.Dsygvd(ibtype, 'V', uplo, n, z, bb, d, work, nwork, &iwork, liwork); err != nil || iinfo != 0 {
						t.Fail()
						fmt.Printf(" ddrvsg2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsygvd(V,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
						if iinfo < 0 {
							return
						} else {
							result.Set(ntest-1, ulpinv)
							goto label100
						}
					}

					//                 Do Test
					dsgt01(ibtype, uplo, n, n, a, b, z, d, work, result.Off(ntest-1))

					//                 Test Dsygvx
					ntest = ntest + 1

					golapack.Dlacpy(Full, n, n, a, ab)
					golapack.Dlacpy(uplo, n, n, b, bb)

					if m, iinfo, err = golapack.Dsygvx(ibtype, 'V', 'A', uplo, n, ab, bb, vl, vu, il, iu, abstol, d, z, work, nwork, toSlice(&iwork, n), &iwork); err != nil || iinfo != 0 {
						t.Fail()
						fmt.Printf(" ddrvsg2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsygvx(V,A"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
						if iinfo < 0 {
							return
						} else {
							result.Set(ntest-1, ulpinv)
							goto label100
						}
					}

					//                 Do Test
					dsgt01(ibtype, uplo, n, n, a, b, z, d, work, result.Off(ntest-1))

					ntest = ntest + 1

					golapack.Dlacpy(Full, n, n, a, ab)
					golapack.Dlacpy(uplo, n, n, b, bb)

					//                 since we do not know the exact eigenvalues of this
					//                 eigenpair, we just set VL and VU as constants.
					//                 It is quite possible that there are no eigenvalues
					//                 in this interval.
					vl = zero
					vu = anorm
					if m, iinfo, err = golapack.Dsygvx(ibtype, 'V', 'V', uplo, n, ab, bb, vl, vu, il, iu, abstol, d, z, work, nwork, toSlice(&iwork, n), &iwork); err != nil || iinfo != 0 {
						t.Fail()
						fmt.Printf(" ddrvsg2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsygvx(V,V,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
						if iinfo < 0 {
							return
						} else {
							result.Set(ntest-1, ulpinv)
							goto label100
						}
					}

					//                 Do Test
					dsgt01(ibtype, uplo, n, m, a, b, z, d, work, result.Off(ntest-1))

					ntest = ntest + 1

					golapack.Dlacpy(Full, n, n, a, ab)
					golapack.Dlacpy(uplo, n, n, b, bb)

					if m, iinfo, err = golapack.Dsygvx(ibtype, 'V', 'I', uplo, n, ab, bb, vl, vu, il, iu, abstol, d, z, work, nwork, toSlice(&iwork, n), &iwork); err != nil || iinfo != 0 {
						t.Fail()
						fmt.Printf(" ddrvsg2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsygvx(V,I,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
						if iinfo < 0 {
							return
						} else {
							result.Set(ntest-1, ulpinv)
							goto label100
						}
					}

					//                 Do Test
					dsgt01(ibtype, uplo, n, m, a, b, z, d, work, result.Off(ntest-1))

				label100:
					;

					//                 Test Dspgv
					ntest = ntest + 1

					//                 Copy the matrices into packed storage.
					if uplo == Upper {
						ij = 1
						for j = 1; j <= n; j++ {
							for i = 1; i <= j; i++ {
								ap.Set(ij-1, a.Get(i-1, j-1))
								bp.Set(ij-1, b.Get(i-1, j-1))
								ij = ij + 1
							}
						}
					} else {
						ij = 1
						for j = 1; j <= n; j++ {
							for i = j; i <= n; i++ {
								ap.Set(ij-1, a.Get(i-1, j-1))
								bp.Set(ij-1, b.Get(i-1, j-1))
								ij = ij + 1
							}
						}
					}

					if iinfo, err = golapack.Dspgv(ibtype, 'V', uplo, n, ap, bp, d, z, work); err != nil || iinfo != 0 {
						t.Fail()
						fmt.Printf(" ddrvsg2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dspgv(V,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
						if iinfo < 0 {
							return
						} else {
							result.Set(ntest-1, ulpinv)
							goto label310
						}
					}

					//                 Do Test
					dsgt01(ibtype, uplo, n, n, a, b, z, d, work, result.Off(ntest-1))

					//                 Test Dspgvd
					ntest = ntest + 1

					//                 Copy the matrices into packed storage.
					if uplo == Upper {
						ij = 1
						for j = 1; j <= n; j++ {
							for i = 1; i <= j; i++ {
								ap.Set(ij-1, a.Get(i-1, j-1))
								bp.Set(ij-1, b.Get(i-1, j-1))
								ij = ij + 1
							}
						}
					} else {
						ij = 1
						for j = 1; j <= n; j++ {
							for i = j; i <= n; i++ {
								ap.Set(ij-1, a.Get(i-1, j-1))
								bp.Set(ij-1, b.Get(i-1, j-1))
								ij = ij + 1
							}
						}
					}

					if iinfo, err = golapack.Dspgvd(ibtype, 'V', uplo, n, ap, bp, d, z, work, nwork, &iwork, liwork); err != nil || iinfo != 0 {
						t.Fail()
						fmt.Printf(" ddrvsg2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dspgvd(V,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
						if iinfo < 0 {
							return
						} else {
							result.Set(ntest-1, ulpinv)
							goto label310
						}
					}

					//                 Do Test
					dsgt01(ibtype, uplo, n, n, a, b, z, d, work, result.Off(ntest-1))

					//                 Test Dspgvx
					ntest = ntest + 1

					//                 Copy the matrices into packed storage.
					if uplo == Upper {
						ij = 1
						for j = 1; j <= n; j++ {
							for i = 1; i <= j; i++ {
								ap.Set(ij-1, a.Get(i-1, j-1))
								bp.Set(ij-1, b.Get(i-1, j-1))
								ij = ij + 1
							}
						}
					} else {
						ij = 1
						for j = 1; j <= n; j++ {
							for i = j; i <= n; i++ {
								ap.Set(ij-1, a.Get(i-1, j-1))
								bp.Set(ij-1, b.Get(i-1, j-1))
								ij = ij + 1
							}
						}
					}

					if m, iinfo, err = golapack.Dspgvx(ibtype, 'V', 'A', uplo, n, ap, bp, vl, vu, il, iu, abstol, d, z, work, toSlice(&iwork, n), &iwork); err != nil || iinfo != 0 {
						t.Fail()
						fmt.Printf(" ddrvsg2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dspgvx(V,A"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
						if iinfo < 0 {
							return
						} else {
							result.Set(ntest-1, ulpinv)
							goto label310
						}
					}

					//                 Do Test
					dsgt01(ibtype, uplo, n, m, a, b, z, d, work, result.Off(ntest-1))

					ntest = ntest + 1

					//                 Copy the matrices into packed storage.
					if uplo == Upper {
						ij = 1
						for j = 1; j <= n; j++ {
							for i = 1; i <= j; i++ {
								ap.Set(ij-1, a.Get(i-1, j-1))
								bp.Set(ij-1, b.Get(i-1, j-1))
								ij = ij + 1
							}
						}
					} else {
						ij = 1
						for j = 1; j <= n; j++ {
							for i = j; i <= n; i++ {
								ap.Set(ij-1, a.Get(i-1, j-1))
								bp.Set(ij-1, b.Get(i-1, j-1))
								ij = ij + 1
							}
						}
					}

					vl = zero
					vu = anorm
					if m, iinfo, err = golapack.Dspgvx(ibtype, 'V', 'V', uplo, n, ap, bp, vl, vu, il, iu, abstol, d, z, work, toSlice(&iwork, n), &iwork); err != nil || iinfo != 0 {
						t.Fail()
						fmt.Printf(" ddrvsg2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dspgvx(V,V"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
						if iinfo < 0 {
							return
						} else {
							result.Set(ntest-1, ulpinv)
							goto label310
						}
					}

					//                 Do Test
					dsgt01(ibtype, uplo, n, m, a, b, z, d, work, result.Off(ntest-1))

					ntest = ntest + 1

					//                 Copy the matrices into packed storage.
					if uplo == Upper {
						ij = 1
						for j = 1; j <= n; j++ {
							for i = 1; i <= j; i++ {
								ap.Set(ij-1, a.Get(i-1, j-1))
								bp.Set(ij-1, b.Get(i-1, j-1))
								ij = ij + 1
							}
						}
					} else {
						ij = 1
						for j = 1; j <= n; j++ {
							for i = j; i <= n; i++ {
								ap.Set(ij-1, a.Get(i-1, j-1))
								bp.Set(ij-1, b.Get(i-1, j-1))
								ij = ij + 1
							}
						}
					}

					if m, iinfo, err = golapack.Dspgvx(ibtype, 'V', 'I', uplo, n, ap, bp, vl, vu, il, iu, abstol, d, z, work, toSlice(&iwork, n), &iwork); err != nil || iinfo != 0 {
						t.Fail()
						fmt.Printf(" ddrvsg2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dspgvx(V,I"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
						if iinfo < 0 {
							return
						} else {
							result.Set(ntest-1, ulpinv)
							goto label310
						}
					}

					//                 Do Test
					dsgt01(ibtype, uplo, n, m, a, b, z, d, work, result.Off(ntest-1))

				label310:
					;

					if ibtype == 1 {
						//                    TEST Dsbgv
						ntest = ntest + 1

						//                    Copy the matrices into band storage.
						if uplo == Upper {
							for j = 1; j <= n; j++ {
								for i = max(1, j-ka); i <= j; i++ {
									ab.Set(ka+1+i-j-1, j-1, a.Get(i-1, j-1))
								}
								for i = max(1, j-kb); i <= j; i++ {
									bb.Set(kb+1+i-j-1, j-1, b.Get(i-1, j-1))
								}
							}
						} else {
							for j = 1; j <= n; j++ {
								for i = j; i <= min(n, j+ka); i++ {
									ab.Set(1+i-j-1, j-1, a.Get(i-1, j-1))
								}
								for i = j; i <= min(n, j+kb); i++ {
									bb.Set(1+i-j-1, j-1, b.Get(i-1, j-1))
								}
							}
						}

						if iinfo, err = golapack.Dsbgv('V', uplo, n, ka, kb, ab, bb, d, z, work); err != nil || iinfo != 0 {
							t.Fail()
							fmt.Printf(" ddrvsg2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsbgv(V,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
							if iinfo < 0 {
								return
							} else {
								result.Set(ntest-1, ulpinv)
								goto label620
							}
						}

						//                    Do Test
						dsgt01(ibtype, uplo, n, n, a, b, z, d, work, result.Off(ntest-1))

						//                    TEST Dsbgvd
						ntest = ntest + 1

						//                    Copy the matrices into band storage.
						if uplo == Upper {
							for j = 1; j <= n; j++ {
								for i = max(1, j-ka); i <= j; i++ {
									ab.Set(ka+1+i-j-1, j-1, a.Get(i-1, j-1))
								}
								for i = max(1, j-kb); i <= j; i++ {
									bb.Set(kb+1+i-j-1, j-1, b.Get(i-1, j-1))
								}
							}
						} else {
							for j = 1; j <= n; j++ {
								for i = j; i <= min(n, j+ka); i++ {
									ab.Set(1+i-j-1, j-1, a.Get(i-1, j-1))
								}
								for i = j; i <= min(n, j+kb); i++ {
									bb.Set(1+i-j-1, j-1, b.Get(i-1, j-1))
								}
							}
						}

						if iinfo, err = golapack.Dsbgvd('V', uplo, n, ka, kb, ab, bb, d, z, work, nwork, &iwork, liwork); err != nil || iinfo != 0 {
							t.Fail()
							fmt.Printf(" ddrvsg2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsbgvd(V,"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
							if iinfo < 0 {
								return
							} else {
								result.Set(ntest-1, ulpinv)
								goto label620
							}
						}

						//                    Do Test
						dsgt01(ibtype, uplo, n, n, a, b, z, d, work, result.Off(ntest-1))

						//                    Test Dsbgvx
						ntest = ntest + 1

						//                    Copy the matrices into band storage.
						if uplo == Upper {
							for j = 1; j <= n; j++ {
								for i = max(1, j-ka); i <= j; i++ {
									ab.Set(ka+1+i-j-1, j-1, a.Get(i-1, j-1))
								}
								for i = max(1, j-kb); i <= j; i++ {
									bb.Set(kb+1+i-j-1, j-1, b.Get(i-1, j-1))
								}
							}
						} else {
							for j = 1; j <= n; j++ {
								for i = j; i <= min(n, j+ka); i++ {
									ab.Set(1+i-j-1, j-1, a.Get(i-1, j-1))
								}
								for i = j; i <= min(n, j+kb); i++ {
									bb.Set(1+i-j-1, j-1, b.Get(i-1, j-1))
								}
							}
						}

						if m, iinfo, err = golapack.Dsbgvx('V', 'A', uplo, n, ka, kb, ab, bb, bp.Matrix(max(1, n), opts), vl, vu, il, iu, abstol, d, z, work, toSlice(&iwork, n), &iwork); err != nil || iinfo != 0 {
							t.Fail()
							fmt.Printf(" ddrvsg2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsbgvx(V,A"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
							if iinfo < 0 {
								return
							} else {
								result.Set(ntest-1, ulpinv)
								goto label620
							}
						}

						//                    Do Test
						dsgt01(ibtype, uplo, n, m, a, b, z, d, work, result.Off(ntest-1))

						ntest = ntest + 1

						//                    Copy the matrices into band storage.
						if uplo == Upper {
							for j = 1; j <= n; j++ {
								for i = max(1, j-ka); i <= j; i++ {
									ab.Set(ka+1+i-j-1, j-1, a.Get(i-1, j-1))
								}
								for i = max(1, j-kb); i <= j; i++ {
									bb.Set(kb+1+i-j-1, j-1, b.Get(i-1, j-1))
								}
							}
						} else {
							for j = 1; j <= n; j++ {
								for i = j; i <= min(n, j+ka); i++ {
									ab.Set(1+i-j-1, j-1, a.Get(i-1, j-1))
								}
								for i = j; i <= min(n, j+kb); i++ {
									bb.Set(1+i-j-1, j-1, b.Get(i-1, j-1))
								}
							}
						}

						vl = zero
						vu = anorm
						if m, iinfo, err = golapack.Dsbgvx('V', 'V', uplo, n, ka, kb, ab, bb, bp.Matrix(max(1, n), opts), vl, vu, il, iu, abstol, d, z, work, toSlice(&iwork, n), &iwork); err != nil || iinfo != 0 {
							t.Fail()
							fmt.Printf(" ddrvsg2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsbgvx(V,V"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
							if iinfo < 0 {
								return
							} else {
								result.Set(ntest-1, ulpinv)
								goto label620
							}
						}

						//                    Do Test
						dsgt01(ibtype, uplo, n, m, a, b, z, d, work, result.Off(ntest-1))

						ntest = ntest + 1

						//                    Copy the matrices into band storage.
						if uplo == Upper {
							for j = 1; j <= n; j++ {
								for i = max(1, j-ka); i <= j; i++ {
									ab.Set(ka+1+i-j-1, j-1, a.Get(i-1, j-1))
								}
								for i = max(1, j-kb); i <= j; i++ {
									bb.Set(kb+1+i-j-1, j-1, b.Get(i-1, j-1))
								}
							}
						} else {
							for j = 1; j <= n; j++ {
								for i = j; i <= min(n, j+ka); i++ {
									ab.Set(1+i-j-1, j-1, a.Get(i-1, j-1))
								}
								for i = j; i <= min(n, j+kb); i++ {
									bb.Set(1+i-j-1, j-1, b.Get(i-1, j-1))
								}
							}
						}

						if m, iinfo, err = golapack.Dsbgvx('V', 'I', uplo, n, ka, kb, ab, bb, bp.Matrix(max(1, n), opts), vl, vu, il, iu, abstol, d, z, work, toSlice(&iwork, n), &iwork); err != nil || iinfo != 0 {
							t.Fail()
							fmt.Printf(" ddrvsg2stg: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Dsbgvx(V,I"+string(uplo.Byte())+")", iinfo, n, jtype, ioldsd)
							if iinfo < 0 {
								return
							} else {
								result.Set(ntest-1, ulpinv)
								goto label620
							}
						}

						//                    Do Test
						dsgt01(ibtype, uplo, n, m, a, b, z, d, work, result.Off(ntest-1))

					}

				label620:
				}
			}

			//           End of Loop -- Check for RESULT(j) > THRESH
			ntestt = ntestt + ntest
			err = dlafts("Dsg", n, n, jtype, ntest, result, ioldsd, thresh, nerrs)
		label640:
		}
	}

	//     Summary
	dlasum("Dsg", nerrs, ntestt)

	return
}
