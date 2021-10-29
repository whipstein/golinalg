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

// ddrvbd checks the singular value decomposition (SVD) drivers
// Dgesvd, Dgesdd, Dgesvdq, Dgesvj, DGEJSV, and DGESVDX.
//
// Both Dgesvd and Dgesdd factor A = U diag(S) VT, where U and VT are
// orthogonal and diag(S) is diagonal with the entries of the array S
// on its diagonal. The entries of S are the singular values,
// nonnegative and stored in decreasing order.  U and VT can be
// optionally not computed, overwritten on A, or computed partially.
//
// A is M by N. Let MNMIN = min( M, N ). S has dimension MNMIN.
// U can be M by M or M by MNMIN. VT can be N by N or MNMIN by N.
//
// When ddrvbd is called, a number of matrix "sizes" (M's and N's)
// and a number of matrix "types" are specified.  For each size (M,N)
// and each type of matrix, and for the minimal workspace as well as
// workspace adequate to permit blocking, an  M x N  matrix "A" will be
// generated and used to test the SVD routines.  For each matrix, A will
// be factored as A = U diag(S) VT and the following 12 tests computed:
//
// Test for Dgesvd:
//
// (1)    | A - U diag(S) VT | / ( |A| max(M,N) ulp )
//
// (2)    | I - U'U | / ( M ulp )
//
// (3)    | I - VT VT' | / ( N ulp )
//
// (4)    S contains MNMIN nonnegative values in decreasing order.
//        (Return 0 if true, 1/ULP if false.)
//
// (5)    | U - Upartial | / ( M ulp ) where Upartial is a partially
//        computed U.
//
// (6)    | VT - VTpartial | / ( N ulp ) where VTpartial is a partially
//        computed VT.
//
// (7)    | S - Spartial | / ( MNMIN ulp |S| ) where Spartial is the
//        vector of singular values from the partial SVD
//
// Test for Dgesdd:
//
// (8)    | A - U diag(S) VT | / ( |A| max(M,N) ulp )
//
// (9)    | I - U'U | / ( M ulp )
//
// (10)   | I - VT VT' | / ( N ulp )
//
// (11)   S contains MNMIN nonnegative values in decreasing order.
//        (Return 0 if true, 1/ULP if false.)
//
// (12)   | U - Upartial | / ( M ulp ) where Upartial is a partially
//        computed U.
//
// (13)   | VT - VTpartial | / ( N ulp ) where VTpartial is a partially
//        computed VT.
//
// (14)   | S - Spartial | / ( MNMIN ulp |S| ) where Spartial is the
//        vector of singular values from the partial SVD
//
// Test for Dgesvdq:
//
// (36)   | A - U diag(S) VT | / ( |A| max(M,N) ulp )
//
// (37)   | I - U'U | / ( M ulp )
//
// (38)   | I - VT VT' | / ( N ulp )
//
// (39)   S contains MNMIN nonnegative values in decreasing order.
//        (Return 0 if true, 1/ULP if false.)
//
// Test for Dgesvj:
//
// (15)   | A - U diag(S) VT | / ( |A| max(M,N) ulp )
//
// (16)   | I - U'U | / ( M ulp )
//
// (17)   | I - VT VT' | / ( N ulp )
//
// (18)   S contains MNMIN nonnegative values in decreasing order.
//        (Return 0 if true, 1/ULP if false.)
//
// Test for DGEJSV:
//
// (19)   | A - U diag(S) VT | / ( |A| max(M,N) ulp )
//
// (20)   | I - U'U | / ( M ulp )
//
// (21)   | I - VT VT' | / ( N ulp )
//
// (22)   S contains MNMIN nonnegative values in decreasing order.
//        (Return 0 if true, 1/ULP if false.)
//
// Test for DGESVDX( 'V', 'V', 'A' )/DGESVDX( 'N', 'N', 'A' )
//
// (23)   | A - U diag(S) VT | / ( |A| max(M,N) ulp )
//
// (24)   | I - U'U | / ( M ulp )
//
// (25)   | I - VT VT' | / ( N ulp )
//
// (26)   S contains MNMIN nonnegative values in decreasing order.
//        (Return 0 if true, 1/ULP if false.)
//
// (27)   | U - Upartial | / ( M ulp ) where Upartial is a partially
//        computed U.
//
// (28)   | VT - VTpartial | / ( N ulp ) where VTpartial is a partially
//        computed VT.
//
// (29)   | S - Spartial | / ( MNMIN ulp |S| ) where Spartial is the
//        vector of singular values from the partial SVD
//
// Test for DGESVDX( 'V', 'V', 'I' )
//
// (30)   | U' A VT''' - diag(S) | / ( |A| max(M,N) ulp )
//
// (31)   | I - U'U | / ( M ulp )
//
// (32)   | I - VT VT' | / ( N ulp )
//
// Test for DGESVDX( 'V', 'V', 'V' )
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
// DOTYPE( 1:NTYPES ); if DOTYPE(j) is .TRUE., then matrix type "j"
// will be generated.
// Currently, the list of possible types is:
//
// (1)  The zero matrix.
// (2)  The identity matrix.
// (3)  A matrix of the form  U D V, where U and V are orthogonal and
//      D has evenly spaced entries 1, ..., ULP with random signs
//      on the diagonal.
// (4)  Same as (3), but multiplied by the underflow-threshold / ULP.
// (5)  Same as (3), but multiplied by the overflow-threshold * ULP.
func ddrvbd(nsizes int, mm, nn []int, ntypes int, dotype []bool, iseed []int, thresh float64, a, u, vt, asav, usav, vtsav *mat.Matrix, s, ssav, e, work *mat.Vector, lwork int, iwork []int, nout int, t *testing.T) (err error) {
	var badmm, badnn bool
	var jobq, jobu, jobvt, _range byte
	var anorm, dif, div, half, one, ovfl, rtunfl, two, ulp, ulpinv, unfl, vl, vu, zero float64
	var i, iinfo, ijq, iju, ijvt, il, itemp, iu, iws, iwtmp, j, jsize, jtype, liwork, lrwork, lswork, m, maxtyp, minwrk, mmax, mnmax, mnmin, mtypes, n, nfail, nmax, ns, nsi, nsv, ntest int

	cjob := make([]byte, 4)
	cjobr := make([]byte, 3)
	cjobv := make([]byte, 2)
	result := vf(39)
	rwork := vf(2)
	ioldsd := make([]int, 4)
	iseed2 := make([]int, 4)

	zero = 0.0
	one = 1.0
	two = 2.0
	half = 0.5
	maxtyp = 5

	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	cjob[0], cjob[1], cjob[2], cjob[3] = 'N', 'O', 'S', 'A'
	cjobr[0], cjobr[1], cjobr[2] = 'A', 'V', 'I'
	cjobv[0], cjobv[1] = 'N', 'V'

	//     Check for errors
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
		minwrk = max(minwrk, max(3*min(mm[j-1], nn[j-1])+max(mm[j-1], nn[j-1]), 5*min(mm[j-1], nn[j-1]-4))+2*int(math.Pow(float64(min(mm[j-1], nn[j-1])), 2)))
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
		gltest.Xerbla2("ddrvbd", err)
		return
	}

	//     Initialize constants
	path := "Dbd"
	nfail = 0
	ntest = 0
	unfl = golapack.Dlamch(SafeMinimum)
	ovfl = one / unfl
	unfl, ovfl = golapack.Dlabad(unfl, ovfl)
	ulp = golapack.Dlamch(Precision)
	rtunfl = math.Sqrt(unfl)
	ulpinv = one / ulp
	(*infot) = 0

	//     Loop over sizes, types
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
				goto label230
			}

			for j = 1; j <= 4; j++ {
				ioldsd[j-1] = iseed[j-1]
			}

			//           Compute "A"
			if mtypes > maxtyp {
				goto label30
			}

			if jtype == 1 {
				//              Zero matrix
				golapack.Dlaset(Full, m, n, zero, zero, a)

			} else if jtype == 2 {
				//              Identity matrix
				golapack.Dlaset(Full, m, n, zero, one, a)

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
				if iinfo, err = matgen.Dlatms(m, n, 'U', &iseed, 'N', s, 4, float64(mnmin), anorm, m-1, n-1, 'N', a, work); iinfo != 0 {
					t.Fail()
					fmt.Printf(" ddrvbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, iseed=%5d\n", "Generator", iinfo, m, n, jtype, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					return
				}
			}

		label30:
			;
			golapack.Dlacpy(Full, m, n, a, asav)

			//           Do for minimal and adequate (for blocking) workspace
			for iws = 1; iws <= 4; iws++ {

				for j = 1; j <= 32; j++ {
					result.Set(j-1, -one)
				}

				//              Test Dgesvd: Factorize A
				iwtmp = max(3*min(m, n)+max(m, n), 5*min(m, n))
				lswork = iwtmp + (iws-1)*(lwork-iwtmp)/3
				lswork = min(lswork, lwork)
				lswork = max(lswork, 1)
				if iws == 4 {
					lswork = lwork
				}

				if iws > 1 {
					golapack.Dlacpy(Full, m, n, asav, a)
				}
				*srnamt = "Dgesvd"
				if iinfo, err = golapack.Dgesvd('A', 'A', m, n, a, ssav, usav, vtsav, work, lswork); err != nil || iinfo != 0 {
					t.Fail()
					fmt.Printf(" ddrvbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, lswork=%6d\n         iseed=%5d\n", "Dgesvd", iinfo, m, n, jtype, lswork, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					return
				}

				//              Do tests 1--4
				result.Set(0, dbdt01(m, n, 0, asav, usav, ssav, e, vtsav, work))
				if m != 0 && n != 0 {
					result.Set(1, dort01('C', m, m, usav, work, lwork))
					result.Set(2, dort01('R', n, n, vtsav, work, lwork))
				}
				result.Set(3, zero)
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
							goto label70
						}
						jobu = cjob[iju]
						jobvt = cjob[ijvt]
						golapack.Dlacpy(Full, m, n, asav, a)
						*srnamt = "Dgesvd"
						if iinfo, err = golapack.Dgesvd(jobu, jobvt, m, n, a, s, u, vt, work, lswork); err != nil {
							panic(err)
						}

						//                    Compare U
						dif = zero
						if m > 0 && n > 0 {
							if iju == 1 {
								dif, iinfo = dort03('C', m, mnmin, m, mnmin, usav, a, work, lwork)
							} else if iju == 2 {
								dif, iinfo = dort03('C', m, mnmin, m, mnmin, usav, u, work, lwork)
							} else if iju == 3 {
								dif, iinfo = dort03('C', m, m, m, mnmin, usav, u, work, lwork)
							}
						}
						result.Set(4, math.Max(result.Get(4), dif))

						//                    Compare VT
						dif = zero
						if m > 0 && n > 0 {
							if ijvt == 1 {
								dif, iinfo = dort03('R', n, mnmin, n, mnmin, vtsav, a, work, lwork)
							} else if ijvt == 2 {
								dif, iinfo = dort03('R', n, mnmin, n, mnmin, vtsav, vt, work, lwork)
							} else if ijvt == 3 {
								dif, iinfo = dort03('R', n, n, n, mnmin, vtsav, vt, work, lwork)
							}
						}
						result.Set(5, math.Max(result.Get(5), dif))

						//                    Compare S
						dif = zero
						div = math.Max(float64(mnmin)*ulp*s.Get(0), unfl)
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
					label70:
					}
				}

				//              Test Dgesdd: Factorize A
				iwtmp = 5*mnmin*mnmin + 9*mnmin + max(m, n)
				lswork = iwtmp + (iws-1)*(lwork-iwtmp)/3
				lswork = min(lswork, lwork)
				lswork = max(lswork, 1)
				if iws == 4 {
					lswork = lwork
				}

				golapack.Dlacpy(Full, m, n, asav, a)
				*srnamt = "Dgesdd"
				if iinfo, err = golapack.Dgesdd('A', m, n, a, ssav, usav, vtsav, work, lswork, &iwork); iinfo != 0 || err != nil {
					t.Fail()
					fmt.Printf(" ddrvbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, lswork=%6d\n         iseed=%5d\n", "Dgesdd", iinfo, m, n, jtype, lswork, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					return
				}

				//              Do tests 8--11
				result.Set(7, dbdt01(m, n, 0, asav, usav, ssav, e, vtsav, work))
				if m != 0 && n != 0 {
					result.Set(8, dort01('C', m, m, usav, work, lwork))
					result.Set(9, dort01('R', n, n, vtsav, work, lwork))
				}
				result.Set(10, zero)
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
					golapack.Dlacpy(Full, m, n, asav, a)
					*srnamt = "Dgesdd"
					if iinfo, err = golapack.Dgesdd(jobq, m, n, a, s, u, vt, work, lswork, &iwork); err != nil {
						panic(err)
					}

					//                 Compare U
					dif = zero
					if m > 0 && n > 0 {
						if ijq == 1 {
							if m >= n {
								dif, _ = dort03('C', m, mnmin, m, mnmin, usav, a, work, lwork)
							} else {
								dif, _ = dort03('C', m, mnmin, m, mnmin, usav, u, work, lwork)
							}
						} else if ijq == 2 {
							dif, _ = dort03('C', m, mnmin, m, mnmin, usav, u, work, lwork)
						}
					}
					result.Set(11, math.Max(result.Get(11), dif))

					//                 Compare VT
					dif = zero
					if m > 0 && n > 0 {
						if ijq == 1 {
							if m >= n {
								dif, _ = dort03('R', n, mnmin, n, mnmin, vtsav, vt, work, lwork)
							} else {
								dif, _ = dort03('R', n, mnmin, n, mnmin, vtsav, a, work, lwork)
							}
						} else if ijq == 2 {
							dif, _ = dort03('R', n, mnmin, n, mnmin, vtsav, vt, work, lwork)
						}
					}
					result.Set(12, math.Max(result.Get(12), dif))

					//                 Compare S
					dif = zero
					div = math.Max(float64(mnmin)*ulp*s.Get(0), unfl)
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

				//              Test Dgesvdq
				//              Note: Dgesvdq only works for M >= N
				result.Set(35, zero)
				result.Set(36, zero)
				result.Set(37, zero)
				result.Set(38, zero)

				if m >= n {
					iwtmp = 5*mnmin*mnmin + 9*mnmin + max(m, n)
					lswork = iwtmp + (iws-1)*(lwork-iwtmp)/3
					lswork = min(lswork, lwork)
					lswork = max(lswork, 1)
					if iws == 4 {
						lswork = lwork
					}

					golapack.Dlacpy(Full, m, n, asav, a)
					*srnamt = "Dgesvdq"

					lrwork = 2
					liwork = max(n, 1)
					if _, iinfo, err = golapack.Dgesvdq('H', 'N', 'N', 'A', 'A', m, n, a, ssav, usav, vtsav, &iwork, liwork, work, lwork, rwork, lrwork); iinfo != 0 || err != nil {
						t.Fail()
						fmt.Printf(" ddrvbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, lswork=%6d\n         iseed=%5d\n", "Dgesvdq", iinfo, m, n, jtype, lswork, ioldsd)
						err = fmt.Errorf("iinfo=%v", abs(iinfo))
						return
					}

					//                 Do tests 36--39
					result.Set(35, dbdt01(m, n, 0, asav, usav, ssav, e, vtsav, work))
					if m != 0 && n != 0 {
						result.Set(36, dort01('C', m, m, usav, work, lwork))
						result.Set(37, dort01('R', n, n, vtsav, work, lwork))
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

				//              Test Dgesvj
				//              Note: Dgesvj only works for M >= N
				result.Set(14, zero)
				result.Set(15, zero)
				result.Set(16, zero)
				result.Set(17, zero)

				if m >= n {
					iwtmp = 5*mnmin*mnmin + 9*mnmin + max(m, n)
					lswork = iwtmp + (iws-1)*(lwork-iwtmp)/3
					lswork = min(lswork, lwork)
					lswork = max(lswork, 1)
					if iws == 4 {
						lswork = lwork
					}

					golapack.Dlacpy(Full, m, n, asav, usav)
					*srnamt = "Dgesvj"
					if _, err = golapack.Dgesvj('G', 'U', 'V', m, n, usav, ssav, 0, a, work, lwork); err != nil {
						panic(err)
					}

					//                 Dgesvj returns V not VT
					for j = 1; j <= n; j++ {
						for i = 1; i <= n; i++ {
							vtsav.Set(j-1, i-1, a.Get(i-1, j-1))
						}
					}

					if iinfo != 0 {
						t.Fail()
						fmt.Printf(" ddrvbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, lswork=%6d\n         iseed=%5d\n", "Dgesvj", iinfo, m, n, jtype, lswork, ioldsd)
						err = fmt.Errorf("iinfo=%v", abs(iinfo))
						return
					}

					//                 Do tests 15--18
					result.Set(14, dbdt01(m, n, 0, asav, usav, ssav, e, vtsav, work))
					if m != 0 && n != 0 {
						result.Set(15, dort01('C', m, m, usav, work, lwork))
						result.Set(16, dort01('R', n, n, vtsav, work, lwork))
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

				//              Test DGEJSV
				//              Note: DGEJSV only works for M >= N
				result.Set(18, zero)
				result.Set(19, zero)
				result.Set(20, zero)
				result.Set(21, zero)
				if m >= n {
					iwtmp = 5*mnmin*mnmin + 9*mnmin + max(m, n)
					lswork = iwtmp + (iws-1)*(lwork-iwtmp)/3
					lswork = min(lswork, lwork)
					lswork = max(lswork, 1)
					if iws == 4 {
						lswork = lwork
					}

					golapack.Dlacpy(Full, m, n, asav, vtsav)
					*srnamt = "DGEJSV"
					iinfo, err = golapack.Dgejsv('G', 'U', 'V', 'R', 'N', 'N', m, n, vtsav, ssav, usav, a, work, lwork, &iwork)

					//                 DGEJSV returns V not VT
					for j = 1; j <= n; j++ {
						for i = 1; i <= n; i++ {
							vtsav.Set(j-1, i-1, a.Get(i-1, j-1))
						}
					}

					if iinfo != 0 {
						t.Fail()
						fmt.Printf(" ddrvbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, lswork=%6d\n         iseed=%5d\n", "Dgejsv", iinfo, m, n, jtype, lswork, ioldsd)
						err = fmt.Errorf("iinfo=%v", abs(iinfo))
						return
					}

					//                 Do tests 19--22
					result.Set(18, dbdt01(m, n, 0, asav, usav, ssav, e, vtsav, work))
					if m != 0 && n != 0 {
						result.Set(19, dort01('C', m, m, usav, work, lwork))
						result.Set(20, dort01('R', n, n, vtsav, work, lwork))
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

				//              Test DGESVDX
				golapack.Dlacpy(Full, m, n, asav, a)
				if ns, iinfo, err = golapack.Dgesvdx('V', 'V', 'A', m, n, a, vl, vu, il, iu, ssav, usav, vtsav, work, lwork, &iwork); iinfo != 0 || err != nil {
					t.Fail()
					fmt.Printf(" ddrvbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, lswork=%6d\n         iseed=%5d\n", "Dgesvdx", iinfo, m, n, jtype, lswork, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					return
				}

				//              Do tests 23--29
				result.Set(22, zero)
				result.Set(23, zero)
				result.Set(24, zero)
				result.Set(22, dbdt01(m, n, 0, asav, usav, ssav, e, vtsav, work))
				if m != 0 && n != 0 {
					result.Set(23, dort01('C', m, m, usav, work, lwork))
					result.Set(24, dort01('R', n, n, vtsav, work, lwork))
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
							goto label170
						}
						jobu = cjobv[iju]
						jobvt = cjobv[ijvt]
						_range = cjobr[0]
						golapack.Dlacpy(Full, m, n, asav, a)
						if ns, iinfo, err = golapack.Dgesvdx(jobu, jobvt, _range, m, n, a, vl, vu, il, iu, s, u, vt, work, lwork, &iwork); err != nil {
							panic(err)
						}

						//                    Compare U
						dif = zero
						if m > 0 && n > 0 {
							if iju == 1 {
								dif, iinfo = dort03('C', m, mnmin, m, mnmin, usav, u, work, lwork)
							}
						}
						result.Set(26, math.Max(result.Get(26), dif))

						//                    Compare VT
						dif = zero
						if m > 0 && n > 0 {
							if ijvt == 1 {
								dif, iinfo = dort03('R', n, mnmin, n, mnmin, vtsav, vt, work, lwork)
							}
						}
						result.Set(27, math.Max(result.Get(27), dif))

						//                    Compare S
						dif = zero
						div = math.Max(float64(mnmin)*ulp*s.Get(0), unfl)
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
					label170:
					}
				}

				//              Do tests 30--32: DGESVDX( 'V', 'V', 'I' )
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
				golapack.Dlacpy(Full, m, n, asav, a)
				if nsi, iinfo, err = golapack.Dgesvdx('V', 'V', 'I', m, n, a, vl, vu, il, iu, s, u, vt, work, lwork, &iwork); iinfo != 0 || err != nil {
					t.Fail()
					fmt.Printf(" ddrvbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, lswork=%6d\n         iseed=%5d\n", "Dgesvdx", iinfo, m, n, jtype, lswork, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					return
				}

				result.Set(29, zero)
				result.Set(30, zero)
				result.Set(31, zero)
				result.Set(29, dbdt05(m, n, asav, s, nsi, u, vt, work))
				result.Set(30, dort01('C', m, nsi, u, work, lwork))
				result.Set(31, dort01('R', nsi, n, vt, work, lwork))

				//              Do tests 33--35: DGESVDX( 'V', 'V', 'V' )
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
				golapack.Dlacpy(Full, m, n, asav, a)
				if nsv, iinfo, err = golapack.Dgesvdx('V', 'V', 'V', m, n, a, vl, vu, il, iu, s, u, vt, work, lwork, &iwork); iinfo != 0 || err != nil {
					t.Fail()
					fmt.Printf(" ddrvbd: %s returned info=%6d.\n         m=%6d, n=%6d, jtype=%6d, lswork=%6d\n         iseed=%5d\n", "Dgesvdx", iinfo, m, n, jtype, lswork, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					return
				}

				result.Set(32, zero)
				result.Set(33, zero)
				result.Set(34, zero)
				result.Set(32, dbdt05(m, n, asav, s, nsv, u, vt, work))
				result.Set(33, dort01('C', m, nsv, u, work, lwork))
				result.Set(34, dort01('R', nsv, n, vt, work, lwork))

				//              End of Loop -- Check for RESULT(j) > THRESH
				for j = 1; j <= 39; j++ {
					if result.Get(j-1) >= thresh {
						t.Fail()
						if nfail == 0 {
							fmt.Printf(" SVD -- Real Singular Value Decomposition Driver \n Matrix types (see ddrvbd for details):\n\n 1 = Zero matrix\n 2 = Identity matrix\n 3 = Evenly spaced singular values near 1\n 4 = Evenly spaced singular values near underflow\n 5 = Evenly spaced singular values near overflow\n\n Tests performed: ( A is dense, U and V are orthogonal,\n                    S is an array, and Upartial, VTpartial, and\n                    Spartial are partially computed U, VT and S),\n\n")
							fmt.Printf(" 1 = | A - U diag(S) VT | / ( |A| max(M,N) ulp ) \n 2 = | I - U**T U | / ( M ulp ) \n 3 = | I - VT VT**T | / ( N ulp ) \n 4 = 0 if S contains min(M,N) nonnegative values in decreasing order, else 1/ulp\n 5 = | U - Upartial | / ( M ulp )\n 6 = | VT - VTpartial | / ( N ulp )\n 7 = | S - Spartial | / ( min(M,N) ulp |S| )\n 8 = | A - U diag(S) VT | / ( |A| max(M,N) ulp ) \n 9 = | I - U**T U | / ( M ulp ) \n10 = | I - VT VT**T | / ( N ulp ) \n11 = 0 if S contains min(M,N) nonnegative values in decreasing order, else 1/ulp\n12 = | U - Upartial | / ( M ulp )\n13 = | VT - VTpartial | / ( N ulp )\n14 = | S - Spartial | / ( min(M,N) ulp |S| )\n15 = | A - U diag(S) VT | / ( |A| max(M,N) ulp ) \n16 = | I - U**T U | / ( M ulp ) \n17 = | I - VT VT**T | / ( N ulp ) \n18 = 0 if S contains min(M,N) nonnegative values in decreasing order, else 1/ulp\n19 = | U - Upartial | / ( M ulp )\n20 = | VT - VTpartial | / ( N ulp )\n21 = | S - Spartial | / ( min(M,N) ulp |S| )\n22 = 0 if S contains min(M,N) nonnegative values in decreasing order, else 1/ulp\n23 = | A - U diag(S) VT | / ( |A| max(M,N) ulp ), DGESVDX(V,V,A) \n24 = | I - U**T U | / ( M ulp ) \n25 = | I - VT VT**T | / ( N ulp ) \n26 = 0 if S contains min(M,N) nonnegative values in decreasing order, else 1/ulp\n27 = | U - Upartial | / ( M ulp )\n28 = | VT - VTpartial | / ( N ulp )\n29 = | S - Spartial | / ( min(M,N) ulp |S| )\n30 = | U**T A VT**T - diag(S) | / ( |A| max(M,N) ulp ), DGESVDX(V,V,I) \n31 = | I - U**T U | / ( M ulp ) \n32 = | I - VT VT**T | / ( N ulp ) \n33 = | U**T A VT**T - diag(S) | / ( |A| max(M,N) ulp ), DGESVDX(V,V,V) \n34 = | I - U**T U | / ( M ulp ) \n35 = | I - VT VT**T | / ( N ulp )  Dgesvdq(H,N,N,A,A\n36 = | A - U diag(S) VT | / ( |A| max(M,N) ulp ) \n37 = | I - U**T U | / ( M ulp ) \n38 = | I - VT VT**T | / ( N ulp ) \n39 = 0 if S contains min(M,N) nonnegative values in decreasing order, else 1/ulp\n\n\n")
						}
						fmt.Printf(" m=%5d, n=%5d, type %1d, IWS=%1d, seed=%4d, test(%2d)=%11.4f\n", m, n, jtype, iws, ioldsd, j, result.Get(j-1))
						nfail++
					}
				}
				ntest = ntest + 39
			}
		label230:
		}
	}

	//     Summary
	alasvm(path, nfail, ntest, 0)

	return
}
