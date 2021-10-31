package eig

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// zdrvbd checks the singular value decomposition (SVD) driver Zgesvd,
// Zgesdd, Zgesvj, Zgejsv, Zgesvdx, and Zgesvdq.
//
// Zgesvd and Zgesdd factors A = U diag(S) VT, where U and VT are
// unitary and diag(S) is diagonal with the entries of the array S on
// its diagonal. The entries of S are the singular values, nonnegative
// and stored in decreasing order.  U and VT can be optionally not
// computed, overwritten on A, or computed partially.
//
// A is M by N. Let MNMIN = min( M, N ). S has dimension MNMIN.
// U can be M by M or M by MNMIN. VT can be N by N or MNMIN by N.
//
// When zdrvbd is called, a number of matrix "sizes" (M's and N's)
// and a number of matrix "types" are specified.  For each size (M,N)
// and each _type of matrix, and for the minimal workspace as well as
// workspace adequate to permit blocking, an  M x N  matrix "A" will be
// generated and used to test the SVD routines.  For each matrix, A will
// be factored as A = U diag(S) VT and the following 12 tests computed:
//
// Test for Zgesvd:
//
// (1)   | A - U diag(S) VT | / ( |A| max(M,N) ulp )
//
// (2)   | I - U'U | / ( M ulp )
//
// (3)   | I - VT VT' | / ( N ulp )
//
// (4)   S contains MNMIN nonnegative values in decreasing order.
//       (Return 0 if true, 1/ULP if false.)
//
// (5)   | U - Upartial | / ( M ulp ) where Upartial is a partially
//       computed U.
//
// (6)   | VT - VTpartial | / ( N ulp ) where VTpartial is a partially
//       computed VT.
//
// (7)   | S - Spartial | / ( MNMIN ulp |S| ) where Spartial is the
//       vector of singular values from the partial SVD
//
// Test for Zgesdd:
//
// (8)   | A - U diag(S) VT | / ( |A| max(M,N) ulp )
//
// (9)   | I - U'U | / ( M ulp )
//
// (10)  | I - VT VT' | / ( N ulp )
//
// (11)  S contains MNMIN nonnegative values in decreasing order.
//       (Return 0 if true, 1/ULP if false.)
//
// (12)  | U - Upartial | / ( M ulp ) where Upartial is a partially
//       computed U.
//
// (13)  | VT - VTpartial | / ( N ulp ) where VTpartial is a partially
//       computed VT.
//
// (14)  | S - Spartial | / ( MNMIN ulp |S| ) where Spartial is the
//       vector of singular values from the partial SVD
//
// Test for Zgesvdq:
//
// (36)  | A - U diag(S) VT | / ( |A| max(M,N) ulp )
//
// (37)  | I - U'U | / ( M ulp )
//
// (38)  | I - VT VT' | / ( N ulp )
//
// (39)  S contains MNMIN nonnegative values in decreasing order.
//       (Return 0 if true, 1/ULP if false.)
//
// Test for Zgesvj:
//
// (15)  | A - U diag(S) VT | / ( |A| max(M,N) ulp )
//
// (16)  | I - U'U | / ( M ulp )
//
// (17)  | I - VT VT' | / ( N ulp )
//
// (18)  S contains MNMIN nonnegative values in decreasing order.
//       (Return 0 if true, 1/ULP if false.)
//
// Test for Zgejsv:
//
// (19)  | A - U diag(S) VT | / ( |A| max(M,N) ulp )
//
// (20)  | I - U'U | / ( M ulp )
//
// (21)  | I - VT VT' | / ( N ulp )
//
// (22)  S contains MNMIN nonnegative values in decreasing order.
//        (Return 0 if true, 1/ULP if false.)
//
// Test for Zgesvdx( 'V', 'V', 'A' )/Zgesvdx( 'N', 'N', 'A' )
//
// (23)  | A - U diag(S) VT | / ( |A| max(M,N) ulp )
//
// (24)  | I - U'U | / ( M ulp )
//
// (25)  | I - VT VT' | / ( N ulp )
//
// (26)  S contains MNMIN nonnegative values in decreasing order.
//       (Return 0 if true, 1/ULP if false.)
//
// (27)  | U - Upartial | / ( M ulp ) where Upartial is a partially
//       computed U.
//
// (28)  | VT - VTpartial | / ( N ulp ) where VTpartial is a partially
//       computed VT.
//
// (29)  | S - Spartial | / ( MNMIN ulp |S| ) where Spartial is the
//       vector of singular values from the partial SVD
//
// Test for Zgesvdx( 'V', 'V', 'I' )
//
// (30)  | U' A VT''' - diag(S) | / ( |A| max(M,N) ulp )
//
// (31)  | I - U'U | / ( M ulp )
//
// (32)  | I - VT VT' | / ( N ulp )
//
// Test for Zgesvdx( 'V', 'V', 'V' )
//
// (33)   | U' A VT''' - diag(S) | / ( |A| max(M,N) ulp )
//
// (34)   | I - U'U | / ( M ulp )
//
// (35)   | I - VT VT' | / ( N ulp )
//
// The "sizes" are specified by the arrays MM(1:NSIZES) and
// NN(1:NSIZES); the value of each element pair (MM(j),NN(j))
// specifies one size.  The "types" are specified by a logical array
// DOTYPE( 1:NTYPES ); if DOTYPE(j) is .TRUE., then matrix _type "j"
// will be generated.
// Currently, the list of possible types is:
//
// (1)  The zero matrix.
// (2)  The identity matrix.
// (3)  A matrix of the form  U D V, where U and V are unitary and
//      D has evenly spaced entries 1, ..., ULP with random signs
//      on the diagonal.
// (4)  Same as (3), but multiplied by the underflow-threshold / ULP.
// (5)  Same as (3), but multiplied by the overflow-threshold * ULP.
func zdrvbd(nsizes int, mm, nn []int, ntypes int, dotype []bool, iseed []int, thresh float64, a, u, vt, asav, usav, vtsav *mat.CMatrix, s, ssav, e *mat.Vector, work *mat.CVector, lwork int, rwork *mat.Vector, iwork []int) (nerrs, ntestt int, err error) {
	var badmm, badnn bool
	var jobq, jobu, jobvt byte
	var cone, czero complex128
	var anorm, dif, div, half, one, ovfl, rtunfl, two, ulp, ulpinv, unfl, vl, vu, zero float64
	var i, iinfo, ijq, iju, ijvt, il, itemp, iu, iwspc, iwtmp, j, jsize, jtype, liwork, lrwork, lswork, m, maxtyp, minwrk, mmax, mnmax, mnmin, mtypes, n, nfail, nmax, ns, nsi, nsv, ntest, ntestf int
	cjob := []byte{'N', 'O', 'S', 'A'}
	cjobv := []byte{'N', 'V'}
	result := vf(39)
	ioldsd := make([]int, 4)
	iseed2 := make([]int, 4)

	zero = 0.0
	one = 1.0
	two = 2.0
	half = 0.5
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	maxtyp = 5
	srnamt := &gltest.Common.Srnamc.Srnamt

	//     Important constants
	nerrs = 0
	ntestt = 0
	ntestf = 0
	badmm = false
	badnn = false
	mmax = 1
	nmax = 1
	mnmax = 1
	minwrk = 1
	for j = 1; j <= nsizes; j++ {
		mmax = max(mmax, mm[j-1])
		if mm[j-1] < 0 {
			badmm = true
		}
		nmax = max(nmax, nn[j-1])
		if nn[j-1] < 0 {
			badnn = true
		}
		mnmax = max(mnmax, min(mm[j-1], nn[j-1]))
		minwrk = max(minwrk, max(3*min(mm[j-1], nn[j-1])+pow(max(mm[j-1], nn[j-1]), 2), 5*min(mm[j-1], nn[j-1]), 3*max(mm[j-1], nn[j-1])))
	}

	//     Check for errors
	if nsizes < 0 {
		err = fmt.Errorf("nsizes < 0: nsizes=%v", nsizes)
	} else if badmm {
		err = fmt.Errorf("badmm: mm=%v", mm)
	} else if badnn {
		err = fmt.Errorf("badnn: nn=%v", nn)
	} else if ntypes < 0 {
		err = fmt.Errorf("ntypes < 0: ntypes=%v", ntypes)
	} else if a.Rows < max(1, mmax) {
		err = fmt.Errorf("a.Rows < max(1, mmax): a.Rows=%v, mmax=%v", a.Rows, mmax)
	} else if u.Rows < max(1, mmax) {
		err = fmt.Errorf("u.Rows < max(1, mmax): u.Rows=%v, mmax=%v", u.Rows, mmax)
	} else if vt.Rows < max(1, nmax) {
		err = fmt.Errorf("vt.Rows < max(1, nmax): vt.Rows=%v, nmax=%v", vt.Rows, nmax)
	} else if minwrk > lwork {
		err = fmt.Errorf("minwrk > lwork: minwrk=%v, lwork=%v", minwrk, lwork)
	}

	if err != nil {
		gltest.Xerbla2("zdrvbd", err)
		return
	}

	//     Quick return if nothing to do
	if nsizes == 0 || ntypes == 0 {
		return
	}

	//     More Important constants
	unfl = golapack.Dlamch(SafeMinimum)
	ovfl = one / unfl
	ulp = golapack.Dlamch(Epsilon)
	ulpinv = one / ulp
	rtunfl = math.Sqrt(unfl)

	//     Loop over sizes, types
	nerrs = 0

	for jsize = 1; jsize <= nsizes; jsize++ {
		m = mm[jsize-1]
		n = nn[jsize-1]
		mnmin = min(m, n)

		if nsizes != 1 {
			mtypes = min(maxtyp, ntypes)
		} else {
			mtypes = min(maxtyp+1, ntypes)
		}

		for jtype = 1; jtype <= mtypes; jtype++ {
			if !dotype[jtype-1] {
				goto label220
			}
			ntest = 0

			for j = 1; j <= 4; j++ {
				ioldsd[j-1] = iseed[j-1]
			}

			//           Compute "A"
			if mtypes > maxtyp {
				goto label50
			}

			if jtype == 1 {
				//              Zero matrix
				golapack.Zlaset(Full, m, n, czero, czero, a)
				for i = 1; i <= min(m, n); i++ {
					s.Set(i-1, zero)
				}

			} else if jtype == 2 {
				//              Identity matrix
				golapack.Zlaset(Full, m, n, czero, cone, a)
				for i = 1; i <= min(m, n); i++ {
					s.Set(i-1, one)
				}

			} else {
				//              (Scaled) random matrix
				if jtype == 3 {
					anorm = one
				}
				if jtype == 4 {
					anorm = unfl / ulp
				}
				if jtype == 5 {
					anorm = ovfl * ulp
				}
				if err = matgen.Zlatms(m, n, 'U', &iseed, 'N', s, 4, float64(mnmin), anorm, m-1, n-1, 'N', a, work); err != nil {
					fmt.Printf(" zdrvbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, iseed=%5d\n", "Generator", iinfo, m, n, jtype, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					return
				}
			}

		label50:
			;
			golapack.Zlacpy(Full, m, n, a, asav)

			//           Do for minimal and adequate (for blocking) workspace
			for iwspc = 1; iwspc <= 4; iwspc++ {
				//              Test for Zgesvd
				iwtmp = 2*min(m, n) + max(m, n)
				lswork = iwtmp + (iwspc-1)*(lwork-iwtmp)/3
				lswork = min(lswork, lwork)
				lswork = max(lswork, 1)
				if iwspc == 4 {
					lswork = lwork
				}

				for j = 1; j <= 35; j++ {
					result.Set(j-1, -one)
				}

				//              Factorize A
				if iwspc > 1 {
					golapack.Zlacpy(Full, m, n, asav, a)
				}
				*srnamt = "Zgesvd"
				if iinfo, err = golapack.Zgesvd('A', 'A', m, n, a, ssav, usav, vtsav, work, lswork, rwork); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, lswork=%6d\n         iseed=%5d\n", "gesvd", iinfo, m, n, jtype, lswork, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					return
				}

				//              Do tests 1--4
				result.Set(0, zbdt01(m, n, 0, asav, usav, ssav, e, vtsav, work, rwork))
				if m != 0 && n != 0 {
					result.Set(1, zunt01('C', mnmin, m, usav, work, lwork, rwork))
					result.Set(2, zunt01('R', mnmin, n, vtsav, work, lwork, rwork))
				}
				result.Set(3, 0)
				for i = 1; i <= mnmin-1; i++ {
					if ssav.Get(i-1) < ssav.Get(i) {
						result.Set(3, ulpinv)
					}
					if ssav.Get(i-1) < zero {
						result.Set(3, ulpinv)
					}
				}
				if mnmin >= 1 {
					if ssav.Get(mnmin-1) < zero {
						result.Set(3, ulpinv)
					}
				}

				//              Do partial SVDs, comparing to SSAV, USAV, and VTSAV
				result.Set(4, zero)
				result.Set(5, zero)
				result.Set(6, zero)
				for iju = 0; iju <= 3; iju++ {
					for ijvt = 0; ijvt <= 3; ijvt++ {
						if (iju == 3 && ijvt == 3) || (iju == 1 && ijvt == 1) {
							goto label90
						}
						jobu = cjob[iju]
						jobvt = cjob[ijvt]
						golapack.Zlacpy(Full, m, n, asav, a)
						*srnamt = "Zgesvd"
						if iinfo, err = golapack.Zgesvd(jobu, jobvt, m, n, a, s, u, vt, work, lswork, rwork); err != nil {
							panic(err)
						}

						//                    Compare U
						dif = zero
						if m > 0 && n > 0 {
							if iju == 1 {
								if dif, err = zunt03('C', m, mnmin, m, mnmin, usav, a, work, lwork, rwork); err != nil {
									panic(err)
								}
							} else if iju == 2 {
								if dif, err = zunt03('C', m, mnmin, m, mnmin, usav, u, work, lwork, rwork); err != nil {
									panic(err)
								}
							} else if iju == 3 {
								if dif, err = zunt03('C', m, m, m, mnmin, usav, u, work, lwork, rwork); err != nil {
									panic(err)
								}
							}
						}
						result.Set(4, math.Max(result.Get(4), dif))

						//                    Compare VT
						dif = zero
						if m > 0 && n > 0 {
							if ijvt == 1 {
								if dif, err = zunt03('R', n, mnmin, n, mnmin, vtsav, a, work, lwork, rwork); err != nil {
									panic(err)
								}
							} else if ijvt == 2 {
								if dif, err = zunt03('R', n, mnmin, n, mnmin, vtsav, vt, work, lwork, rwork); err != nil {
									panic(err)
								}
							} else if ijvt == 3 {
								if dif, err = zunt03('R', n, n, n, mnmin, vtsav, vt, work, lwork, rwork); err != nil {
									panic(err)
								}
							}
						}
						result.Set(5, math.Max(result.Get(5), dif))

						//                    Compare S
						dif = zero
						div = math.Max(float64(mnmin)*ulp*s.Get(0), golapack.Dlamch(SafeMinimum))
						for i = 1; i <= mnmin-1; i++ {
							if ssav.Get(i-1) < ssav.Get(i) {
								dif = ulpinv
							}
							if ssav.Get(i-1) < zero {
								dif = ulpinv
							}
							dif = math.Max(dif, math.Abs(ssav.Get(i-1)-s.Get(i-1))/div)
						}
						result.Set(6, math.Max(result.Get(6), dif))
					label90:
					}
				}

				//              Test for Zgesdd
				iwtmp = 2*mnmin*mnmin + 2*mnmin + max(m, n)
				lswork = iwtmp + (iwspc-1)*(lwork-iwtmp)/3
				lswork = min(lswork, lwork)
				lswork = max(lswork, 1)
				if iwspc == 4 {
					lswork = lwork
				}

				//              Factorize A
				golapack.Zlacpy(Full, m, n, asav, a)
				*srnamt = "Zgesdd"
				if iinfo, err = golapack.Zgesdd('A', m, n, a, ssav, usav, vtsav, work, lswork, rwork, &iwork); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, lswork=%6d\n         iseed=%5d\n", "Zgesdd", iinfo, m, n, jtype, lswork, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					return
				}

				//              Do tests 1--4
				result.Set(7, zbdt01(m, n, 0, asav, usav, ssav, e, vtsav, work, rwork))
				if m != 0 && n != 0 {
					result.Set(8, zunt01('C', mnmin, m, usav, work, lwork, rwork))
					result.Set(9, zunt01('R', mnmin, n, vtsav, work, lwork, rwork))
				}
				result.Set(10, 0)
				for i = 1; i <= mnmin-1; i++ {
					if ssav.Get(i-1) < ssav.Get(i) {
						result.Set(10, ulpinv)
					}
					if ssav.Get(i-1) < zero {
						result.Set(10, ulpinv)
					}
				}
				if mnmin >= 1 {
					if ssav.Get(mnmin-1) < zero {
						result.Set(10, ulpinv)
					}
				}

				//              Do partial SVDs, comparing to SSAV, USAV, and VTSAV
				result.Set(11, zero)
				result.Set(12, zero)
				result.Set(13, zero)
				for ijq = 0; ijq <= 2; ijq++ {
					jobq = cjob[ijq]
					golapack.Zlacpy(Full, m, n, asav, a)
					*srnamt = "Zgesdd"
					if iinfo, err = golapack.Zgesdd(jobq, m, n, a, s, u, vt, work, lswork, rwork, &iwork); err != nil {
						panic(err)
					}

					//                 Compare U
					dif = zero
					if m > 0 && n > 0 {
						if ijq == 1 {
							if m >= n {
								if dif, err = zunt03('C', m, mnmin, m, mnmin, usav, a, work, lwork, rwork); err != nil {
									panic(err)
								}
							} else {
								if dif, err = zunt03('C', m, mnmin, m, mnmin, usav, u, work, lwork, rwork); err != nil {
									panic(err)
								}
							}
						} else if ijq == 2 {
							if dif, err = zunt03('C', m, mnmin, m, mnmin, usav, u, work, lwork, rwork); err != nil {
								panic(err)
							}
						}
					}
					result.Set(11, math.Max(result.Get(11), dif))

					//                 Compare VT
					dif = zero
					if m > 0 && n > 0 {
						if ijq == 1 {
							if m >= n {
								if dif, err = zunt03('R', n, mnmin, n, mnmin, vtsav, vt, work, lwork, rwork); err != nil {
									panic(err)
								}
							} else {
								if dif, err = zunt03('R', n, mnmin, n, mnmin, vtsav, a, work, lwork, rwork); err != nil {
									panic(err)
								}
							}
						} else if ijq == 2 {
							if dif, err = zunt03('R', n, mnmin, n, mnmin, vtsav, vt, work, lwork, rwork); err != nil {
								panic(err)
							}
						}
					}
					result.Set(12, math.Max(result.Get(12), dif))

					//                 Compare S
					dif = zero
					div = math.Max(float64(mnmin)*ulp*s.Get(0), golapack.Dlamch(SafeMinimum))
					for i = 1; i <= mnmin-1; i++ {
						if ssav.Get(i-1) < ssav.Get(i) {
							dif = ulpinv
						}
						if ssav.Get(i-1) < zero {
							dif = ulpinv
						}
						dif = math.Max(dif, math.Abs(ssav.Get(i-1)-s.Get(i-1))/div)
					}
					result.Set(13, math.Max(result.Get(13), dif))
				}

				//              Test Zgesvdq
				//              Note: Zgesvdq only works for M >= N
				result.Set(35, zero)
				result.Set(36, zero)
				result.Set(37, zero)
				result.Set(38, zero)

				if m >= n {
					iwtmp = 2*mnmin*mnmin + 2*mnmin + max(m, n)
					lswork = iwtmp + (iwspc-1)*(lwork-iwtmp)/3
					lswork = min(lswork, lwork)
					lswork = max(lswork, 1)
					if iwspc == 4 {
						lswork = lwork
					}

					golapack.Zlacpy(Full, m, n, asav, a)
					*srnamt = "Zgesvdq"

					lrwork = max(2, m, 5*n)
					liwork = max(n, 1)
					if _, lwork, iinfo, err = golapack.Zgesvdq('H', 'N', 'N', 'A', 'A', m, n, a, ssav, usav, vtsav, &iwork, liwork, work, lwork, rwork, lrwork); err != nil || iinfo != 0 {
						fmt.Printf(" zdrvbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, lswork=%6d\n         iseed=%5d\n", "Zgesvdq", iinfo, m, n, jtype, lswork, ioldsd)
						return
					}

					//                 Do tests 36--39
					result.Set(35, zbdt01(m, n, 0, asav, usav, ssav, e, vtsav, work, rwork))
					if m != 0 && n != 0 {
						result.Set(36, zunt01('C', m, m, usav, work, lwork, rwork))
						result.Set(37, zunt01('R', n, n, vtsav, work, lwork, rwork))
					}
					result.Set(38, zero)
					for i = 1; i <= mnmin-1; i++ {
						if ssav.Get(i-1) < ssav.Get(i) {
							result.Set(38, ulpinv)
						}
						if ssav.Get(i-1) < zero {
							result.Set(38, ulpinv)
						}
					}
					if mnmin >= 1 {
						if ssav.Get(mnmin-1) < zero {
							result.Set(38, ulpinv)
						}
					}
				}

				//              Test Zgesvj
				//              Note: Zgesvj only works for M >= N
				result.Set(14, zero)
				result.Set(15, zero)
				result.Set(16, zero)
				result.Set(17, zero)

				if m >= n {
					iwtmp = 2*mnmin*mnmin + 2*mnmin + max(m, n)
					lswork = iwtmp + (iwspc-1)*(lwork-iwtmp)/3
					lswork = min(lswork, lwork)
					lswork = max(lswork, 1)
					lrwork = max(6, n)
					if iwspc == 4 {
						lswork = lwork
					}

					golapack.Zlacpy(Full, m, n, asav, usav)
					*srnamt = "Zgesvj"
					if iinfo, err = golapack.Zgesvj('G', 'U', 'V', m, n, usav, ssav, 0, a, work, lwork, rwork, lrwork); err != nil {
						panic(err)
					}

					//                 Zgesvj returns V not VH
					for j = 1; j <= n; j++ {
						for i = 1; i <= n; i++ {
							vtsav.Set(j-1, i-1, a.GetConj(i-1, j-1))
						}
					}

					if iinfo != 0 {
						fmt.Printf(" zdrvbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, lswork=%6d\n         iseed=%5d\n", "gesvj", iinfo, m, n, jtype, lswork, ioldsd)
						err = fmt.Errorf("iinfo=%v", abs(iinfo))
						return
					}

					//                 Do tests 15--18
					result.Set(14, zbdt01(m, n, 0, asav, usav, ssav, e, vtsav, work, rwork))
					if m != 0 && n != 0 {
						result.Set(15, zunt01('C', m, m, usav, work, lwork, rwork))
						result.Set(16, zunt01('R', n, n, vtsav, work, lwork, rwork))
					}
					result.Set(17, zero)
					for i = 1; i <= mnmin-1; i++ {
						if ssav.Get(i-1) < ssav.Get(i) {
							result.Set(17, ulpinv)
						}
						if ssav.Get(i-1) < zero {
							result.Set(17, ulpinv)
						}
					}
					if mnmin >= 1 {
						if ssav.Get(mnmin-1) < zero {
							result.Set(17, ulpinv)
						}
					}
				}

				//              Test Zgejsv
				//              Note: Zgejsv only works for M >= N
				result.Set(18, zero)
				result.Set(19, zero)
				result.Set(20, zero)
				result.Set(21, zero)
				if m >= n {
					iwtmp = 2*mnmin*mnmin + 2*mnmin + max(m, n)
					lswork = iwtmp + (iwspc-1)*(lwork-iwtmp)/3
					lswork = min(lswork, lwork)
					lswork = max(lswork, 1)
					if iwspc == 4 {
						lswork = lwork
					}
					lrwork = max(7, n+2*m)

					golapack.Zlacpy(Full, m, n, asav, vtsav)
					*srnamt = "Zgejsv"
					iinfo, err = golapack.Zgejsv('G', 'U', 'V', 'R', 'N', 'N', m, n, vtsav, ssav, usav, a, work, lwork, rwork, lrwork, &iwork)

					//                 Zgejsv returns V not VH
					for j = 1; j <= n; j++ {
						for i = 1; i <= n; i++ {
							vtsav.Set(j-1, i-1, a.GetConj(i-1, j-1))
						}
					}

					if err != nil || iinfo != 0 {
						fmt.Printf(" zdrvbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, lswork=%6d\n         iseed=%5d\n", "gejsv", iinfo, m, n, jtype, lswork, ioldsd)
						return
					}

					//                 Do tests 19--22
					result.Set(10, zbdt01(m, n, 0, asav, usav, ssav, e, vtsav, work, rwork))
					if m != 0 && n != 0 {
						result.Set(19, zunt01('C', m, m, usav, work, lwork, rwork))
						result.Set(20, zunt01('R', n, n, vtsav, work, lwork, rwork))
					}
					result.Set(21, zero)
					for i = 1; i <= mnmin-1; i++ {
						if ssav.Get(i-1) < ssav.Get(i) {
							result.Set(21, ulpinv)
						}
						if ssav.Get(i-1) < zero {
							result.Set(21, ulpinv)
						}
					}
					if mnmin >= 1 {
						if ssav.Get(mnmin-1) < zero {
							result.Set(21, ulpinv)
						}
					}
				}

				//              Test Zgesvdx
				//
				//              Factorize A
				golapack.Zlacpy(Full, m, n, asav, a)
				*srnamt = "Zgesvdx"
				if iinfo, err = golapack.Zgesvdx('V', 'V', 'A', m, n, a, vl, vu, il, iu, ns, ssav, usav, vtsav, work, lwork, rwork, &iwork); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, lswork=%6d\n         iseed=%5d\n", "gesvdx", iinfo, m, n, jtype, lswork, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					return
				}

				//              Do tests 1--4
				result.Set(22, zero)
				result.Set(23, zero)
				result.Set(24, zero)
				result.Set(22, zbdt01(m, n, 0, asav, usav, ssav, e, vtsav, work, rwork))
				if m != 0 && n != 0 {
					result.Set(23, zunt01('C', mnmin, m, usav, work, lwork, rwork))
					result.Set(24, zunt01('R', mnmin, n, vtsav, work, lwork, rwork))
				}
				result.Set(25, zero)
				for i = 1; i <= mnmin-1; i++ {
					if ssav.Get(i-1) < ssav.Get(i) {
						result.Set(25, ulpinv)
					}
					if ssav.Get(i-1) < zero {
						result.Set(25, ulpinv)
					}
				}
				if mnmin >= 1 {
					if ssav.Get(mnmin-1) < zero {
						result.Set(25, ulpinv)
					}
				}

				//              Do partial SVDs, comparing to SSAV, USAV, and VTSAV
				result.Set(26, zero)
				result.Set(27, zero)
				result.Set(28, zero)
				for iju = 0; iju <= 1; iju++ {
					for ijvt = 0; ijvt <= 1; ijvt++ {
						if (iju == 0 && ijvt == 0) || (iju == 1 && ijvt == 1) {
							goto label160
						}
						jobu = cjobv[iju]
						jobvt = cjobv[ijvt]
						// _range = cjobr[0]
						golapack.Zlacpy(Full, m, n, asav, a)
						*srnamt = "Zgesvdx"
						if iinfo, err = golapack.Zgesvdx(jobu, jobvt, 'A', m, n, a, vl, vu, il, iu, ns, ssav, u, vt, work, lwork, rwork, &iwork); err != nil {
							panic(err)
						}

						//                    Compare U
						dif = zero
						if m > 0 && n > 0 {
							if iju == 1 {
								if dif, err = zunt03('C', m, mnmin, m, mnmin, usav, u, work, lwork, rwork); err != nil {
									panic(err)
								}
							}
						}
						result.Set(26, math.Max(result.Get(26), dif))

						//                    Compare VT
						dif = zero
						if m > 0 && n > 0 {
							if ijvt == 1 {
								if dif, err = zunt03('R', n, mnmin, n, mnmin, vtsav, vt, work, lwork, rwork); err != nil {
									panic(err)
								}
							}
						}
						result.Set(27, math.Max(result.Get(27), dif))

						//                    Compare S
						dif = zero
						div = math.Max(float64(mnmin)*ulp*s.Get(0), golapack.Dlamch(SafeMinimum))
						for i = 1; i <= mnmin-1; i++ {
							if ssav.Get(i-1) < ssav.Get(i) {
								dif = ulpinv
							}
							if ssav.Get(i-1) < zero {
								dif = ulpinv
							}
							dif = math.Max(dif, math.Abs(ssav.Get(i-1)-s.Get(i-1))/div)
						}
						result.Set(28, math.Max(result.Get(28), dif))
					label160:
					}
				}

				//              Do tests 8--10
				for i = 1; i <= 4; i++ {
					iseed2[i-1] = iseed[i-1]
				}
				if mnmin <= 1 {
					il = 1
					iu = max(1, mnmin)
				} else {
					il = 1 + int(float64(mnmin-1)*matgen.Dlarnd(1, &iseed2))
					iu = 1 + int(float64(mnmin-1)*matgen.Dlarnd(1, &iseed2))
					if iu < il {
						itemp = iu
						iu = il
						il = itemp
					}
				}
				golapack.Zlacpy(Full, m, n, asav, a)
				*srnamt = "Zgesvdx"
				if iinfo, err = golapack.Zgesvdx('V', 'V', 'I', m, n, a, vl, vu, il, iu, nsi, s, u, vt, work, lwork, rwork, &iwork); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, lswork=%6d\n         iseed=%5d\n", "gesvdx", iinfo, m, n, jtype, lswork, ioldsd)
					return
				}

				result.Set(29, zero)
				result.Set(30, zero)
				result.Set(31, zero)
				result.Set(29, zbdt05(m, n, asav, s, nsi, u, vt, work))
				if m != 0 && n != 0 {
					result.Set(30, zunt01('C', m, nsi, u, work, lwork, rwork))
					result.Set(31, zunt01('R', nsi, n, vt, work, lwork, rwork))
				}

				//              Do tests 11--13
				if mnmin > 0 && nsi > 1 {
					if il != 1 {
						vu = ssav.Get(il-1) + math.Max(half*math.Abs(ssav.Get(il-1)-ssav.Get(il-1-1)), math.Max(ulp*anorm, two*rtunfl))
					} else {
						vu = ssav.Get(0) + math.Max(half*math.Abs(ssav.Get(ns-1)-ssav.Get(0)), math.Max(ulp*anorm, two*rtunfl))
					}
					if iu != ns {
						vl = ssav.Get(iu-1) - math.Max(ulp*anorm, math.Max(two*rtunfl, half*math.Abs(ssav.Get(iu)-ssav.Get(iu-1))))
					} else {
						vl = ssav.Get(ns-1) - math.Max(ulp*anorm, math.Max(two*rtunfl, half*math.Abs(ssav.Get(ns-1)-ssav.Get(0))))
					}
					vl = math.Max(vl, zero)
					vu = math.Max(vu, zero)
					if vl >= vu {
						vu = math.Max(vu*2, vu+vl+half)
					}
				} else {
					vl = zero
					vu = one
				}
				golapack.Zlacpy(Full, m, n, asav, a)
				*srnamt = "Zgesvdx"
				if iinfo, err = golapack.Zgesvdx('V', 'V', 'V', m, n, a, vl, vu, il, iu, nsv, s, u, vt, work, lwork, rwork, &iwork); err != nil || iinfo != 0 {
					fmt.Printf(" zdrvbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, lswork=%6d\n         iseed=%5d\n", "gesvdx", iinfo, m, n, jtype, lswork, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					return
				}
				//
				result.Set(32, zero)
				result.Set(33, zero)
				result.Set(34, zero)
				result.Set(32, zbdt05(m, n, asav, s, nsv, u, vt, work))
				if m != 0 && n != 0 {
					result.Set(33, zunt01('C', m, nsv, u, work, lwork, rwork))
					result.Set(34, zunt01('R', nsv, n, vt, work, lwork, rwork))
				}

				//              End of Loop -- Check for RESULT(j) > THRESH
				ntest = 0
				nfail = 0
				for j = 1; j <= 39; j++ {
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
					fmt.Printf(" SVD -- Complex Singular Value Decomposition Driver \n Matrix types (see zdrvbd for details):\n\n 1 = Zero matrix\n 2 = Identity matrix\n 3 = Evenly spaced singular values near 1\n 4 = Evenly spaced singular values near underflow\n 5 = Evenly spaced singular values near overflow\n\n Tests performed: ( A is dense, U and V are unitary,\n                    S is an array, and Upartial, VTpartial, and\n                    Spartial are partially computed U, VT and S),\n\n")
					fmt.Printf(" Tests performed with Test Threshold = %8.2f\n Zgesvd: \n 1 = | A - U diag(S) VT | / ( |A| max(M,N) ulp ) \n 2 = | I - U**T U | / ( M ulp ) \n 3 = | I - VT VT**T | / ( N ulp ) \n 4 = 0 if S contains min(M,N) nonnegative values in decreasing order, else 1/ulp\n 5 = | U - Upartial | / ( M ulp )\n 6 = | VT - VTpartial | / ( N ulp )\n 7 = | S - Spartial | / ( min(M,N) ulp |S| )\n Zgesdd: \n 8 = | A - U diag(S) VT | / ( |A| max(M,N) ulp ) \n 9 = | I - U**T U | / ( M ulp ) \n10 = | I - VT VT**T | / ( N ulp ) \n11 = 0 if S contains min(M,N) nonnegative values in decreasing order, else 1/ulp\n12 = | U - Upartial | / ( M ulp )\n13 = | VT - VTpartial | / ( N ulp )\n14 = | S - Spartial | / ( min(M,N) ulp |S| )\n Zgesvj: \n\n15 = | A - U diag(S) VT | / ( |A| max(M,N) ulp ) \n16 = | I - U**T U | / ( M ulp ) \n17 = | I - VT VT**T | / ( N ulp ) \n18 = 0 if S contains min(M,N) nonnegative values in decreasing order, else 1/ulp\n ZGESJV: \n\n19 = | A - U diag(S) VT | / ( |A| max(M,N) ulp )\n20 = | I - U**T U | / ( M ulp ) \n21 = | I - VT VT**T | / ( N ulp ) \n22 = 0 if S contains min(M,N) nonnegative values in decreasing order, else 1/ulp\n Zgesvdx(V,V,A): \n23 = | A - U diag(S) VT | / ( |A| max(M,N) ulp ) \n24 = | I - U**T U | / ( M ulp ) \n25 = | I - VT VT**T | / ( N ulp ) \n26 = 0 if S contains min(M,N) nonnegative values in decreasing order, else 1/ulp\n27 = | U - Upartial | / ( M ulp )\n28 = | VT - VTpartial | / ( N ulp )\n29 = | S - Spartial | / ( min(M,N) ulp |S| )\n Zgesvdx(V,V,I): \n30 = | U**T A VT**T - diag(S) | / ( |A| max(M,N) ulp )\n31 = | I - U**T U | / ( M ulp ) \n32 = | I - VT VT**T | / ( N ulp ) \n Zgesvdx(V,V,V) \n33 = | U**T A VT**T - diag(S) | / ( |A| max(M,N) ulp )\n34 = | I - U**T U | / ( M ulp ) \n35 = | I - VT VT**T | / ( N ulp )  Zgesvdq(H,N,N,A,A\n36 = | A - U diag(S) VT | / ( |A| max(M,N) ulp ) \n37 = | I - U**T U | / ( M ulp ) \n38 = | I - VT VT**T | / ( N ulp ) \n39 = 0 if S contains min(M,N) nonnegative values in decreasing order, else 1/ulp\n\n\n", thresh)
					ntestf = 2
				}

				for j = 1; j <= 39; j++ {
					if result.Get(j-1) >= thresh {
						fmt.Printf(" m=%5d, n=%5d, _type %1d, iws=%1d, seed=%4d, test(%2d)=%11.4f\n", m, n, jtype, iwspc, ioldsd, j, result.Get(j-1))
						err = fmt.Errorf(" m=%5d, n=%5d, _type %1d, iws=%1d, seed=%4d, test(%2d)=%11.4f\n", m, n, jtype, iwspc, ioldsd, j, result.Get(j-1))
					}
				}

				nerrs = nerrs + nfail
				ntestt = ntestt + ntest

			}

		label220:
		}
	}

	//     Summary
	// alasvm("Zbd", nerrs, ntestt, 0)

	return
}
