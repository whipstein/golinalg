package eig

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// zdrvev checks the nonsymmetric eigenvalue problem driver ZGEEV.
//
//    When zdrvev is called, a number of matrix "sizes" ("n's") and a
//    number of matrix "types" are specified.  For each size ("n")
//    and each _type of matrix, one matrix will be generated and used
//    to test the nonsymmetric eigenroutines.  For each matrix, 7
//    tests will be performed:
//
//    (1)     | A * VR - VR * W | / ( n |A| ulp )
//
//      Here VR is the matrix of unit right eigenvectors.
//      W is a diagonal matrix with diagonal entries W(j).
//
//    (2)     | A**H * VL - VL * W**H | / ( n |A| ulp )
//
//      Here VL is the matrix of unit left eigenvectors, A**H is the
//      conjugate-transpose of A, and W is as above.
//
//    (3)     | |VR(i)| - 1 | / ulp and whether largest component real
//
//      VR(i) denotes the i-th column of VR.
//
//    (4)     | |VL(i)| - 1 | / ulp and whether largest component real
//
//      VL(i) denotes the i-th column of VL.
//
//    (5)     W(full) = W(partial)
//
//      W(full) denotes the eigenvalues computed when both VR and VL
//      are also computed, and W(partial) denotes the eigenvalues
//      computed when only W, only W and VR, or only W and VL are
//      computed.
//
//    (6)     VR(full) = VR(partial)
//
//      VR(full) denotes the right eigenvectors computed when both VR
//      and VL are computed, and VR(partial) denotes the result
//      when only VR is computed.
//
//     (7)     VL(full) = VL(partial)
//
//      VL(full) denotes the left eigenvectors computed when both VR
//      and VL are also computed, and VL(partial) denotes the result
//      when only VL is computed.
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
//    (7)  Same as (4), but multiplied by a constant near
//         the overflow threshold
//    (8)  Same as (4), but multiplied by a constant near
//         the underflow threshold
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
//         SQRT( ULP ) and T has evenly spaced entries 1, ..., ULP
//         with random complex angles on the diagonal and random O(1)
//         entries in the upper triangle.
//
//    (14) A matrix of the form  X' T X, where X has condition
//         SQRT( ULP ) and T has geometrically spaced entries
//         1, ..., ULP with random complex angles on the diagonal
//         and random O(1) entries in the upper triangle.
//
//    (15) A matrix of the form  X' T X, where X has condition
//         SQRT( ULP ) and T has "clustered" entries 1, ULP,..., ULP
//         with random complex angles on the diagonal and random O(1)
//         entries in the upper triangle.
//
//    (16) A matrix of the form  X' T X, where X has condition
//         SQRT( ULP ) and T has complex eigenvalues randomly chosen
//         from ULP < |z| < 1 and random O(1) entries in the upper
//         triangle.
//
//    (17) Same as (16), but multiplied by a constant
//         near the overflow threshold
//    (18) Same as (16), but multiplied by a constant
//         near the underflow threshold
//
//    (19) Nonsymmetric matrix with random entries chosen from |z| < 1
//         If N is at least 4, all entries in first two rows and last
//         row, and first column and last two columns are zero.
//    (20) Same as (19), but multiplied by a constant
//         near the overflow threshold
//    (21) Same as (19), but multiplied by a constant
//         near the underflow threshold
func zdrvev(nsizes int, nn []int, ntypes int, dotype []bool, iseed []int, thresh float64, a, h *mat.CMatrix, w, w1 *mat.CVector, vl, vr, lre *mat.CMatrix, result *mat.Vector, work *mat.CVector, nwork int, rwork *mat.Vector, iwork []int) (nerrs, ntestt int, err error) {
	var badnn bool
	var cone, czero complex128
	var anorm, cond, conds, one, ovfl, rtulp, rtulpi, tnrm, two, ulp, ulpinv, unfl, vmx, vrmx, vtst, zero float64
	var iinfo, imode, itype, iwk, j, jcol, jj, jsize, jtype, maxtyp, mtypes, n, nfail, nmax, nnwork, ntest, ntestf int
	dum := cvf(1)
	res := vf(2)
	idumma := make([]int, 1)
	ioldsd := make([]int, 4)
	kconds := []int{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0}
	kmagn := []int{1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 2, 3}
	kmode := []int{0, 0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 5, 4, 3, 1, 5, 5, 5, 4, 3, 1}
	ktype := []int{1, 2, 3, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 9}

	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	zero = 0.0
	one = 1.0
	two = 2.0
	maxtyp = 21

	path := "Zev"

	//     Check for errors
	ntestt = 0
	ntestf = 0

	//     Important constants
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
	} else if a.Rows < 1 || a.Rows < nmax {
		err = fmt.Errorf("a.Rows < 1 || a.Rows < nmax: a.Rows=%v, nmax=%v", a.Rows, nmax)
	} else if vl.Rows < 1 || vl.Rows < nmax {
		err = fmt.Errorf("vl.Rows < 1 || vl.Rows < nmax: vl.Rows=%v, nmax=%v", vl.Rows, nmax)
	} else if vr.Rows < 1 || vr.Rows < nmax {
		err = fmt.Errorf("vr.Rows < 1 || vr.Rows < nmax: vr.Rows=%v, nmax=%v", vr.Rows, nmax)
	} else if lre.Rows < 1 || lre.Rows < nmax {
		err = fmt.Errorf("lre.Rows < 1 || lre.Rows < nmax: lre.Rows=%v, nmax=%v", lre.Rows, nmax)
	} else if 5*nmax+2*pow(nmax, 2) > nwork {
		err = fmt.Errorf("5*nmax+2*pow(nmax, 2) > nwork: nmax=%v, nwork=%v", nmax, nwork)
	}

	if err != nil {
		gltest.Xerbla2("zdrvev", err)
		return
	}

	//     Quick return if nothing to do
	if nsizes == 0 || ntypes == 0 {
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
				goto label260
			}

			//           Save iseed in case of an error.
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

			golapack.Zlaset(Full, a.Rows, n, czero, czero, a)
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
					a.SetRe(jcol-1, jcol-1, anorm)
				}

			} else if itype == 3 {
				//              Jordan Block
				for jcol = 1; jcol <= n; jcol++ {
					a.SetRe(jcol-1, jcol-1, anorm)
					if jcol > 1 {
						a.Set(jcol-1, jcol-1-1, cone)
					}
				}

			} else if itype == 4 {
				//              Diagonal Matrix, [Eigen]values Specified
				err = matgen.Zlatms(n, n, 'S', &iseed, 'H', rwork, imode, cond, anorm, 0, 0, 'N', a, work.Off(n))

			} else if itype == 5 {
				//              Hermitian, eigenvalues specified
				err = matgen.Zlatms(n, n, 'S', &iseed, 'H', rwork, imode, cond, anorm, n, n, 'N', a, work.Off(n))

			} else if itype == 6 {
				//              General, eigenvalues specified
				if kconds[jtype-1] == 1 {
					conds = one
				} else if kconds[jtype-1] == 2 {
					conds = rtulpi
				} else {
					conds = zero
				}

				err = matgen.Zlatme(n, 'D', &iseed, work, imode, cond, cone, 'T', 'T', 'T', rwork, 4, conds, n, n, anorm, a, work.Off(2*n))

			} else if itype == 7 {
				//              Diagonal, random eigenvalues
				err = matgen.Zlatmr(n, n, 'D', &iseed, 'N', work, 6, one, cone, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, 0, 0, zero, anorm, 'N', a, &iwork)

			} else if itype == 8 {
				//              Symmetric, random eigenvalues
				err = matgen.Zlatmr(n, n, 'D', &iseed, 'H', work, 6, one, cone, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, n, n, zero, anorm, 'N', a, &iwork)

			} else if itype == 9 {
				//              General, random eigenvalues
				err = matgen.Zlatmr(n, n, 'D', &iseed, 'N', work, 6, one, cone, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, n, n, zero, anorm, 'N', a, &iwork)
				if n >= 4 {
					golapack.Zlaset(Full, 2, n, czero, czero, a)
					golapack.Zlaset(Full, n-3, 1, czero, czero, a.Off(2, 0))
					golapack.Zlaset(Full, n-3, 2, czero, czero, a.Off(2, n-1-1))
					golapack.Zlaset(Full, 1, n, czero, czero, a.Off(n-1, 0))
				}

			} else if itype == 10 {
				//              Triangular, random eigenvalues
				err = matgen.Zlatmr(n, n, 'D', &iseed, 'N', work, 6, one, cone, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, n, 0, zero, anorm, 'N', a, &iwork)

			} else {

				iinfo = 1
			}

			if iinfo != 0 || err != nil {
				fmt.Printf(" zdrvev: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Generator", iinfo, n, jtype, ioldsd)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				return
			}

		label90:
			;

			//           Test for minimal and generous workspace
			for iwk = 1; iwk <= 2; iwk++ {
				if iwk == 1 {
					nnwork = 2 * n
				} else {
					nnwork = 5*n + 2*pow(n, 2)
				}
				nnwork = max(nnwork, 1)

				//              Initialize RESULT
				for j = 1; j <= 7; j++ {
					result.Set(j-1, -one)
				}

				//              Compute eigenvalues and eigenvectors, and test them
				golapack.Zlacpy(Full, n, n, a, h)
				if iinfo, err = golapack.Zgeev('V', 'V', n, h, w, vl, vr, work, nnwork, rwork); err != nil || iinfo != 0 {
					result.Set(0, ulpinv)
					fmt.Printf(" zdrvev: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zgeev1", iinfo, n, jtype, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					goto label220
				}

				//              Do Test (1)
				zget22(NoTrans, NoTrans, NoTrans, n, a, vr, w, work, rwork, res)
				result.Set(0, res.Get(0))

				//              Do Test (2)
				zget22(ConjTrans, NoTrans, ConjTrans, n, a, vl, w, work, rwork, res)
				result.Set(1, res.Get(0))

				//              Do Test (3)
				for j = 1; j <= n; j++ {
					tnrm = vr.Off(0, j-1).CVector().Nrm2(n, 1)
					result.Set(2, math.Max(result.Get(2), math.Min(ulpinv, math.Abs(tnrm-one)/ulp)))
					vmx = zero
					vrmx = zero
					for jj = 1; jj <= n; jj++ {
						vtst = vr.GetMag(jj-1, j-1)
						if vtst > vmx {
							vmx = vtst
						}
						if vr.GetIm(jj-1, j-1) == zero && math.Abs(vr.GetRe(jj-1, j-1)) > vrmx {
							vrmx = math.Abs(vr.GetRe(jj-1, j-1))
						}
					}
					if vrmx/vmx < one-two*ulp {
						result.Set(2, ulpinv)
					}
				}

				//              Do Test (4)
				for j = 1; j <= n; j++ {
					tnrm = vl.Off(0, j-1).CVector().Nrm2(n, 1)
					result.Set(3, math.Max(result.Get(3), math.Min(ulpinv, math.Abs(tnrm-one)/ulp)))
					vmx = zero
					vrmx = zero
					for jj = 1; jj <= n; jj++ {
						vtst = vl.GetMag(jj-1, j-1)
						if vtst > vmx {
							vmx = vtst
						}
						if vl.GetIm(jj-1, j-1) == zero && math.Abs(vl.GetRe(jj-1, j-1)) > vrmx {
							vrmx = math.Abs(vl.GetRe(jj-1, j-1))
						}
					}
					if vrmx/vmx < one-two*ulp {
						result.Set(3, ulpinv)
					}
				}

				//              Compute eigenvalues only, and test them
				golapack.Zlacpy(Full, n, n, a, h)
				if iinfo, err = golapack.Zgeev('N', 'N', n, h, w1, dum.CMatrix(1, opts), dum.CMatrix(1, opts), work, nnwork, rwork); err != nil || iinfo != 0 {
					result.Set(0, ulpinv)
					fmt.Printf(" zdrvev: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zgeev2", iinfo, n, jtype, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					goto label220
				}

				//              Do Test (5)
				for j = 1; j <= n; j++ {
					if w.Get(j-1) != w1.Get(j-1) {
						result.Set(4, ulpinv)
					}
				}

				//              Compute eigenvalues and right eigenvectors, and test them
				golapack.Zlacpy(Full, n, n, a, h)
				if iinfo, err = golapack.Zgeev('N', 'V', n, h, w1, dum.CMatrix(1, opts), lre, work, nnwork, rwork); err != nil || iinfo != 0 {
					result.Set(0, ulpinv)
					fmt.Printf(" zdrvev: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zgeev3", iinfo, n, jtype, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					goto label220
				}

				//              Do Test (5) again
				for j = 1; j <= n; j++ {
					if w.Get(j-1) != w1.Get(j-1) {
						result.Set(4, ulpinv)
					}
				}

				//              Do Test (6)
				for j = 1; j <= n; j++ {
					for jj = 1; jj <= n; jj++ {
						if vr.Get(j-1, jj-1) != lre.Get(j-1, jj-1) {
							result.Set(5, ulpinv)
						}
					}
				}

				//              Compute eigenvalues and left eigenvectors, and test them
				golapack.Zlacpy(Full, n, n, a, h)
				if iinfo, err = golapack.Zgeev('V', 'N', n, h, w1, lre, dum.CMatrix(1, opts), work, nnwork, rwork); err != nil || iinfo != 0 {
					result.Set(0, ulpinv)
					fmt.Printf(" zdrvev: %s returned info=%6d.\n         n=%6d, jtype=%6d, iseed=%5d\n", "Zgeev4", iinfo, n, jtype, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					goto label220
				}

				//              Do Test (5) again
				for j = 1; j <= n; j++ {
					if w.Get(j-1) != w1.Get(j-1) {
						result.Set(4, ulpinv)
					}
				}

				//              Do Test (7)
				for j = 1; j <= n; j++ {
					for jj = 1; jj <= n; jj++ {
						if vl.Get(j-1, jj-1) != lre.Get(j-1, jj-1) {
							result.Set(6, ulpinv)
						}
					}
				}

				//              End of Loop -- Check for RESULT(j) > THRESH
			label220:
				;

				ntest = 0
				nfail = 0
				for j = 1; j <= 7; j++ {
					if result.Get(j-1) >= zero {
						ntest = ntest + 1
					}
					if result.Get(j-1) >= thresh {
						nfail++
					}
				}

				if nfail > 0 {
					ntestf = ntestf + 1
				}
				if ntestf == 1 {
					fmt.Printf("\n %3s -- Complex Eigenvalue-Eigenvector Decomposition Driver\n Matrix types (see zdrvev for details): \n", path)
					fmt.Printf("\n Special Matrices:\n  1=Zero matrix.                          5=Diagonal: geometr. spaced entries.\n  2=Identity matrix.                      6=Diagonal: clustered entries.\n  3=Transposed Jordan block.              7=Diagonal: large, evenly spaced.\n  4=Diagonal: evenly spaced entries.      8=Diagonal: small, evenly spaced.\n")
					fmt.Printf(" Dense, Non-Symmetric Matrices:\n  9=Well-cond., evenly spaced eigenvals. 14=Ill-cond., geomet. spaced eigenals.\n 10=Well-cond., geom. spaced eigenvals.  15=Ill-conditioned, clustered e.vals.\n 11=Well-conditioned, clustered e.vals.  16=Ill-cond., random complex       \n 12=Well-cond., random complex           17=Ill-cond., large rand. complx     \n 13=Ill-conditioned, evenly spaced.      18=Ill-cond., small rand. complx     \n")
					fmt.Printf(" 19=Matrix with random O(1) entries.     21=Matrix with small random entries.\n 20=Matrix with large random entries.   \n\n")
					fmt.Printf(" Tests performed with test threshold =%8.2f\n\n 1 = | A VR - VR W | / ( n |A| ulp ) \n 2 = | conj-trans(A) VL - VL conj-trans(W) | / ( n |A| ulp ) \n 3 = | |VR(i)| - 1 | / ulp \n 4 = | |VL(i)| - 1 | / ulp \n 5 = 0 if W same no matter if VR or VL computed, 1/ulp otherwise\n 6 = 0 if VR same no matter if VL computed,  1/ulp otherwise\n 7 = 0 if VL same no matter if VR computed,  1/ulp otherwise\n\n", thresh)
					ntestf = 2
				}

				for j = 1; j <= 7; j++ {
					if result.Get(j-1) >= thresh {
						fmt.Printf(" n=%5d, IWK=%2d, seed=%4d, _type %2d, test(%2d)=%10.3f\n", n, iwk, ioldsd, jtype, j, result.Get(j-1))
						err = fmt.Errorf(" n=%5d, IWK=%2d, seed=%4d, _type %2d, test(%2d)=%10.3f\n", n, iwk, ioldsd, jtype, j, result.Get(j-1))
					}
				}

				nerrs = nerrs + nfail
				ntestt = ntestt + ntest

			}
		label260:
		}
	}

	//     Summary
	// dlasum(path, nerrs, ntestt)

	return
}
