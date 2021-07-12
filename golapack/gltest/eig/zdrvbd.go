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

// Zdrvbd checks the singular value decomposition (SVD) driver ZGESVD,
// ZGESDD, ZGESVJ, ZGEJSV, ZGESVDX, and ZGESVDQ.
//
// ZGESVD and ZGESDD factors A = U diag(S) VT, where U and VT are
// unitary and diag(S) is diagonal with the entries of the array S on
// its diagonal. The entries of S are the singular values, nonnegative
// and stored in decreasing order.  U and VT can be optionally not
// computed, overwritten on A, or computed partially.
//
// A is M by N. Let MNMIN = min( M, N ). S has dimension MNMIN.
// U can be M by M or M by MNMIN. VT can be N by N or MNMIN by N.
//
// When ZDRVBD is called, a number of matrix "sizes" (M's and N's)
// and a number of matrix "types" are specified.  For each size (M,N)
// and each _type of matrix, and for the minimal workspace as well as
// workspace adequate to permit blocking, an  M x N  matrix "A" will be
// generated and used to test the SVD routines.  For each matrix, A will
// be factored as A = U diag(S) VT and the following 12 tests computed:
//
// Test for ZGESVD:
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
// Test for ZGESDD:
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
// Test for ZGESVDQ:
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
// Test for ZGESVJ:
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
// Test for ZGEJSV:
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
// Test for ZGESVDX( 'V', 'V', 'A' )/ZGESVDX( 'N', 'N', 'A' )
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
// Test for ZGESVDX( 'V', 'V', 'I' )
//
// (30)  | U' A VT''' - diag(S) | / ( |A| max(M,N) ulp )
//
// (31)  | I - U'U | / ( M ulp )
//
// (32)  | I - VT VT' | / ( N ulp )
//
// Test for ZGESVDX( 'V', 'V', 'V' )
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
func Zdrvbd(nsizes *int, mm, nn *[]int, ntypes *int, dotype *[]bool, iseed *[]int, thresh *float64, a *mat.CMatrix, lda *int, u *mat.CMatrix, ldu *int, vt *mat.CMatrix, ldvt *int, asav, usav, vtsav *mat.CMatrix, s, ssav, e *mat.Vector, work *mat.CVector, lwork *int, rwork *mat.Vector, iwork *[]int, nounit, info *int, t *testing.T) {
	var badmm, badnn bool
	var jobq, jobu, jobvt byte
	var cone, czero complex128
	var anorm, dif, div, half, one, ovfl, rtunfl, two, ulp, ulpinv, unfl, vl, vu, zero float64
	var i, iinfo, ijq, iju, ijvt, il, itemp, iu, iwspc, iwtmp, j, jsize, jtype, liwork, lrwork, lswork, m, maxtyp, minwrk, mmax, mnmax, mnmin, mtypes, n, nerrs, nfail, nmax, ns, nsi, nsv, ntest, ntestf, ntestt, numrank int
	cjob := make([]byte, 4)
	cjobr := make([]byte, 3)
	cjobv := make([]byte, 2)
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

	cjob[0], cjob[1], cjob[2], cjob[3] = 'N', 'O', 'S', 'A'
	cjobr[0], cjobr[1], cjobr[2] = 'A', 'V', 'I'
	cjobv[0], cjobv[1] = 'N', 'V'

	//     Check for errors
	(*info) = 0

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
	for j = 1; j <= (*nsizes); j++ {
		mmax = max(mmax, (*mm)[j-1])
		if (*mm)[j-1] < 0 {
			badmm = true
		}
		nmax = max(nmax, (*nn)[j-1])
		if (*nn)[j-1] < 0 {
			badnn = true
		}
		mnmax = max(mnmax, min((*mm)[j-1], (*nn)[j-1]))
		minwrk = max(minwrk, max(3*min((*mm)[j-1], (*nn)[j-1])+pow(max((*mm)[j-1], (*nn)[j-1]), 2), 5*min((*mm)[j-1], (*nn)[j-1]), 3*max((*mm)[j-1], (*nn)[j-1])))
	}

	//     Check for errors
	if (*nsizes) < 0 {
		(*info) = -1
	} else if badmm {
		(*info) = -2
	} else if badnn {
		(*info) = -3
	} else if (*ntypes) < 0 {
		(*info) = -4
	} else if (*lda) < max(1, mmax) {
		(*info) = -10
	} else if (*ldu) < max(1, mmax) {
		(*info) = -12
	} else if (*ldvt) < max(1, nmax) {
		(*info) = -14
	} else if minwrk > (*lwork) {
		(*info) = -21
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZDRVBD"), -(*info))
		return
	}

	//     Quick return if nothing to do
	if (*nsizes) == 0 || (*ntypes) == 0 {
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

	for jsize = 1; jsize <= (*nsizes); jsize++ {
		m = (*mm)[jsize-1]
		n = (*nn)[jsize-1]
		mnmin = min(m, n)

		if (*nsizes) != 1 {
			mtypes = min(maxtyp, *ntypes)
		} else {
			mtypes = min(maxtyp+1, *ntypes)
		}

		for jtype = 1; jtype <= mtypes; jtype++ {
			if !(*dotype)[jtype-1] {
				goto label220
			}
			ntest = 0

			for j = 1; j <= 4; j++ {
				ioldsd[j-1] = (*iseed)[j-1]
			}

			//           Compute "A"
			if mtypes > maxtyp {
				goto label50
			}

			if jtype == 1 {
				//              Zero matrix
				golapack.Zlaset('F', &m, &n, &czero, &czero, a, lda)
				for i = 1; i <= min(m, n); i++ {
					s.Set(i-1, zero)
				}

			} else if jtype == 2 {
				//              Identity matrix
				golapack.Zlaset('F', &m, &n, &czero, &cone, a, lda)
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
				matgen.Zlatms(&m, &n, 'U', iseed, 'N', s, func() *int { y := 4; return &y }(), toPtrf64(float64(mnmin)), &anorm, toPtr(m-1), toPtr(n-1), 'N', a, lda, work, &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVBD: %s returned INFO=%6d.\n         M=%6d, N=%6d, JTYPE=%6d, ISEED=%5d\n", "Generator", iinfo, m, n, jtype, ioldsd)
					(*info) = abs(iinfo)
					return
				}
			}

		label50:
			;
			golapack.Zlacpy('F', &m, &n, a, lda, asav, lda)

			//           Do for minimal and adequate (for blocking) workspace
			for iwspc = 1; iwspc <= 4; iwspc++ {
				//              Test for ZGESVD
				iwtmp = 2*min(m, n) + max(m, n)
				lswork = iwtmp + (iwspc-1)*((*lwork)-iwtmp)/3
				lswork = min(lswork, *lwork)
				lswork = max(lswork, 1)
				if iwspc == 4 {
					lswork = (*lwork)
				}

				for j = 1; j <= 35; j++ {
					result.Set(j-1, -one)
				}

				//              Factorize A
				if iwspc > 1 {
					golapack.Zlacpy('F', &m, &n, asav, lda, a, lda)
				}
				*srnamt = "ZGESVD"
				golapack.Zgesvd('A', 'A', &m, &n, a, lda, ssav, usav, ldu, vtsav, ldvt, work, &lswork, rwork, &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVBD: %s returned INFO=%6d.\n         M=%6d, N=%6d, JTYPE=%6d, LSWORK=%6d\n         ISEED=%5d\n", "GESVD", iinfo, m, n, jtype, lswork, ioldsd)
					(*info) = abs(iinfo)
					return
				}

				//              Do tests 1--4
				Zbdt01(&m, &n, func() *int { y := 0; return &y }(), asav, lda, usav, ldu, ssav, e, vtsav, ldvt, work, rwork, result.GetPtr(0))
				if m != 0 && n != 0 {
					Zunt01('C', &mnmin, &m, usav, ldu, work, lwork, rwork, result.GetPtr(1))
					Zunt01('R', &mnmin, &n, vtsav, ldvt, work, lwork, rwork, result.GetPtr(2))
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
						golapack.Zlacpy('F', &m, &n, asav, lda, a, lda)
						*srnamt = "ZGESVD"
						golapack.Zgesvd(jobu, jobvt, &m, &n, a, lda, s, u, ldu, vt, ldvt, work, &lswork, rwork, &iinfo)

						//                    Compare U
						dif = zero
						if m > 0 && n > 0 {
							if iju == 1 {
								Zunt03('C', &m, &mnmin, &m, &mnmin, usav, ldu, a, lda, work, lwork, rwork, &dif, &iinfo)
							} else if iju == 2 {
								Zunt03('C', &m, &mnmin, &m, &mnmin, usav, ldu, u, ldu, work, lwork, rwork, &dif, &iinfo)
							} else if iju == 3 {
								Zunt03('C', &m, &m, &m, &mnmin, usav, ldu, u, ldu, work, lwork, rwork, &dif, &iinfo)
							}
						}
						result.Set(4, math.Max(result.Get(4), dif))

						//                    Compare VT
						dif = zero
						if m > 0 && n > 0 {
							if ijvt == 1 {
								Zunt03('R', &n, &mnmin, &n, &mnmin, vtsav, ldvt, a, lda, work, lwork, rwork, &dif, &iinfo)
							} else if ijvt == 2 {
								Zunt03('R', &n, &mnmin, &n, &mnmin, vtsav, ldvt, vt, ldvt, work, lwork, rwork, &dif, &iinfo)
							} else if ijvt == 3 {
								Zunt03('R', &n, &n, &n, &mnmin, vtsav, ldvt, vt, ldvt, work, lwork, rwork, &dif, &iinfo)
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

				//              Test for ZGESDD
				iwtmp = 2*mnmin*mnmin + 2*mnmin + max(m, n)
				lswork = iwtmp + (iwspc-1)*((*lwork)-iwtmp)/3
				lswork = min(lswork, *lwork)
				lswork = max(lswork, 1)
				if iwspc == 4 {
					lswork = (*lwork)
				}

				//              Factorize A
				golapack.Zlacpy('F', &m, &n, asav, lda, a, lda)
				*srnamt = "ZGESDD"
				golapack.Zgesdd('A', &m, &n, a, lda, ssav, usav, ldu, vtsav, ldvt, work, &lswork, rwork, iwork, &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVBD: %s returned INFO=%6d.\n         M=%6d, N=%6d, JTYPE=%6d, LSWORK=%6d\n         ISEED=%5d\n", "GESDD", iinfo, m, n, jtype, lswork, ioldsd)
					(*info) = abs(iinfo)
					return
				}

				//              Do tests 1--4
				Zbdt01(&m, &n, func() *int { y := 0; return &y }(), asav, lda, usav, ldu, ssav, e, vtsav, ldvt, work, rwork, result.GetPtr(7))
				if m != 0 && n != 0 {
					Zunt01('C', &mnmin, &m, usav, ldu, work, lwork, rwork, result.GetPtr(8))
					Zunt01('R', &mnmin, &n, vtsav, ldvt, work, lwork, rwork, result.GetPtr(9))
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
					golapack.Zlacpy('F', &m, &n, asav, lda, a, lda)
					*srnamt = "ZGESDD"
					golapack.Zgesdd(jobq, &m, &n, a, lda, s, u, ldu, vt, ldvt, work, &lswork, rwork, iwork, &iinfo)

					//                 Compare U
					dif = zero
					if m > 0 && n > 0 {
						if ijq == 1 {
							if m >= n {
								Zunt03('C', &m, &mnmin, &m, &mnmin, usav, ldu, a, lda, work, lwork, rwork, &dif, &iinfo)
							} else {
								Zunt03('C', &m, &mnmin, &m, &mnmin, usav, ldu, u, ldu, work, lwork, rwork, &dif, &iinfo)
							}
						} else if ijq == 2 {
							Zunt03('C', &m, &mnmin, &m, &mnmin, usav, ldu, u, ldu, work, lwork, rwork, &dif, &iinfo)
						}
					}
					result.Set(11, math.Max(result.Get(11), dif))

					//                 Compare VT
					dif = zero
					if m > 0 && n > 0 {
						if ijq == 1 {
							if m >= n {
								Zunt03('R', &n, &mnmin, &n, &mnmin, vtsav, ldvt, vt, ldvt, work, lwork, rwork, &dif, &iinfo)
							} else {
								Zunt03('R', &n, &mnmin, &n, &mnmin, vtsav, ldvt, a, lda, work, lwork, rwork, &dif, &iinfo)
							}
						} else if ijq == 2 {
							Zunt03('R', &n, &mnmin, &n, &mnmin, vtsav, ldvt, vt, ldvt, work, lwork, rwork, &dif, &iinfo)
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

				//              Test ZGESVDQ
				//              Note: ZGESVDQ only works for M >= N
				result.Set(35, zero)
				result.Set(36, zero)
				result.Set(37, zero)
				result.Set(38, zero)

				if m >= n {
					iwtmp = 2*mnmin*mnmin + 2*mnmin + max(m, n)
					lswork = iwtmp + (iwspc-1)*((*lwork)-iwtmp)/3
					lswork = min(lswork, *lwork)
					lswork = max(lswork, 1)
					if iwspc == 4 {
						lswork = (*lwork)
					}

					golapack.Zlacpy('F', &m, &n, asav, lda, a, lda)
					*srnamt = "ZGESVDQ"

					lrwork = max(2, m, 5*n)
					liwork = max(n, 1)
					golapack.Zgesvdq('H', 'N', 'N', 'A', 'A', &m, &n, a, lda, ssav, usav, ldu, vtsav, ldvt, &numrank, iwork, &liwork, work, lwork, rwork, &lrwork, &iinfo)

					if iinfo != 0 {
						t.Fail()
						fmt.Printf(" ZDRVBD: %s returned INFO=%6d.\n         M=%6d, N=%6d, JTYPE=%6d, LSWORK=%6d\n         ISEED=%5d\n", "ZGESVDQ", iinfo, m, n, jtype, lswork, ioldsd)
						(*info) = abs(iinfo)
						return
					}

					//                 Do tests 36--39
					Zbdt01(&m, &n, func() *int { y := 0; return &y }(), asav, lda, usav, ldu, ssav, e, vtsav, ldvt, work, rwork, result.GetPtr(35))
					if m != 0 && n != 0 {
						Zunt01('C', &m, &m, usav, ldu, work, lwork, rwork, result.GetPtr(36))
						Zunt01('R', &n, &n, vtsav, ldvt, work, lwork, rwork, result.GetPtr(37))
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

				//              Test ZGESVJ
				//              Note: ZGESVJ only works for M >= N
				result.Set(14, zero)
				result.Set(15, zero)
				result.Set(16, zero)
				result.Set(17, zero)

				if m >= n {
					iwtmp = 2*mnmin*mnmin + 2*mnmin + max(m, n)
					lswork = iwtmp + (iwspc-1)*((*lwork)-iwtmp)/3
					lswork = min(lswork, *lwork)
					lswork = max(lswork, 1)
					lrwork = max(6, n)
					if iwspc == 4 {
						lswork = (*lwork)
					}

					golapack.Zlacpy('F', &m, &n, asav, lda, usav, lda)
					*srnamt = "ZGESVJ"
					golapack.Zgesvj('G', 'U', 'V', &m, &n, usav, lda, ssav, func() *int { y := 0; return &y }(), a, ldvt, work, lwork, rwork, &lrwork, &iinfo)

					//                 ZGESVJ returns V not VH
					for j = 1; j <= n; j++ {
						for i = 1; i <= n; i++ {
							vtsav.Set(j-1, i-1, a.GetConj(i-1, j-1))
						}
					}

					if iinfo != 0 {
						t.Fail()
						fmt.Printf(" ZDRVBD: %s returned INFO=%6d.\n         M=%6d, N=%6d, JTYPE=%6d, LSWORK=%6d\n         ISEED=%5d\n", "GESVJ", iinfo, m, n, jtype, lswork, ioldsd)
						(*info) = abs(iinfo)
						return
					}

					//                 Do tests 15--18
					Zbdt01(&m, &n, func() *int { y := 0; return &y }(), asav, lda, usav, ldu, ssav, e, vtsav, ldvt, work, rwork, result.GetPtr(14))
					if m != 0 && n != 0 {
						Zunt01('C', &m, &m, usav, ldu, work, lwork, rwork, result.GetPtr(15))
						Zunt01('R', &n, &n, vtsav, ldvt, work, lwork, rwork, result.GetPtr(16))
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

				//              Test ZGEJSV
				//              Note: ZGEJSV only works for M >= N
				result.Set(18, zero)
				result.Set(19, zero)
				result.Set(20, zero)
				result.Set(21, zero)
				if m >= n {
					iwtmp = 2*mnmin*mnmin + 2*mnmin + max(m, n)
					lswork = iwtmp + (iwspc-1)*((*lwork)-iwtmp)/3
					lswork = min(lswork, *lwork)
					lswork = max(lswork, 1)
					if iwspc == 4 {
						lswork = (*lwork)
					}
					lrwork = max(7, n+2*m)

					golapack.Zlacpy('F', &m, &n, asav, lda, vtsav, lda)
					*srnamt = "ZGEJSV"
					golapack.Zgejsv('G', 'U', 'V', 'R', 'N', 'N', &m, &n, vtsav, lda, ssav, usav, ldu, a, ldvt, work, lwork, rwork, &lrwork, iwork, &iinfo)

					//                 ZGEJSV returns V not VH
					for j = 1; j <= n; j++ {
						for i = 1; i <= n; i++ {
							vtsav.Set(j-1, i-1, a.GetConj(i-1, j-1))
						}
					}

					if iinfo != 0 {
						t.Fail()
						fmt.Printf(" ZDRVBD: %s returned INFO=%6d.\n         M=%6d, N=%6d, JTYPE=%6d, LSWORK=%6d\n         ISEED=%5d\n", "GEJSV", iinfo, m, n, jtype, lswork, ioldsd)
						(*info) = abs(iinfo)
						return
					}

					//                 Do tests 19--22
					Zbdt01(&m, &n, func() *int { y := 0; return &y }(), asav, lda, usav, ldu, ssav, e, vtsav, ldvt, work, rwork, result.GetPtr(18))
					if m != 0 && n != 0 {
						Zunt01('C', &m, &m, usav, ldu, work, lwork, rwork, result.GetPtr(19))
						Zunt01('R', &n, &n, vtsav, ldvt, work, lwork, rwork, result.GetPtr(20))
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

				//              Test ZGESVDX
				//
				//              Factorize A
				golapack.Zlacpy('F', &m, &n, asav, lda, a, lda)
				*srnamt = "ZGESVDX"
				golapack.Zgesvdx('V', 'V', 'A', &m, &n, a, lda, &vl, &vu, &il, &iu, &ns, ssav, usav, ldu, vtsav, ldvt, work, lwork, rwork, iwork, &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVBD: %s returned INFO=%6d.\n         M=%6d, N=%6d, JTYPE=%6d, LSWORK=%6d\n         ISEED=%5d\n", "GESVDX", iinfo, m, n, jtype, lswork, ioldsd)
					(*info) = abs(iinfo)
					return
				}

				//              Do tests 1--4
				result.Set(22, zero)
				result.Set(23, zero)
				result.Set(24, zero)
				Zbdt01(&m, &n, func() *int { y := 0; return &y }(), asav, lda, usav, ldu, ssav, e, vtsav, ldvt, work, rwork, result.GetPtr(22))
				if m != 0 && n != 0 {
					Zunt01('C', &mnmin, &m, usav, ldu, work, lwork, rwork, result.GetPtr(23))
					Zunt01('R', &mnmin, &n, vtsav, ldvt, work, lwork, rwork, result.GetPtr(24))
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
						golapack.Zlacpy('F', &m, &n, asav, lda, a, lda)
						*srnamt = "ZGESVDX"
						golapack.Zgesvdx(jobu, jobvt, 'A', &m, &n, a, lda, &vl, &vu, &il, &iu, &ns, ssav, u, ldu, vt, ldvt, work, lwork, rwork, iwork, &iinfo)

						//                    Compare U
						dif = zero
						if m > 0 && n > 0 {
							if iju == 1 {
								Zunt03('C', &m, &mnmin, &m, &mnmin, usav, ldu, u, ldu, work, lwork, rwork, &dif, &iinfo)
							}
						}
						result.Set(26, math.Max(result.Get(26), dif))

						//                    Compare VT
						dif = zero
						if m > 0 && n > 0 {
							if ijvt == 1 {
								Zunt03('R', &n, &mnmin, &n, &mnmin, vtsav, ldvt, vt, ldvt, work, lwork, rwork, &dif, &iinfo)
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
					iseed2[i-1] = (*iseed)[i-1]
				}
				if mnmin <= 1 {
					il = 1
					iu = max(1, mnmin)
				} else {
					il = 1 + int(float64(mnmin-1)*matgen.Dlarnd(func() *int { y := 1; return &y }(), &iseed2))
					iu = 1 + int(float64(mnmin-1)*matgen.Dlarnd(func() *int { y := 1; return &y }(), &iseed2))
					if iu < il {
						itemp = iu
						iu = il
						il = itemp
					}
				}
				golapack.Zlacpy('F', &m, &n, asav, lda, a, lda)
				*srnamt = "ZGESVDX"
				golapack.Zgesvdx('V', 'V', 'I', &m, &n, a, lda, &vl, &vu, &il, &iu, &nsi, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVBD: %s returned INFO=%6d.\n         M=%6d, N=%6d, JTYPE=%6d, LSWORK=%6d\n         ISEED=%5d\n", "GESVDX", iinfo, m, n, jtype, lswork, ioldsd)
					(*info) = abs(iinfo)
					return
				}

				result.Set(29, zero)
				result.Set(30, zero)
				result.Set(31, zero)
				Zbdt05(&m, &n, asav, lda, s, &nsi, u.CVector(0, 0), ldu, vt, ldvt, work, result.GetPtr(29))
				if m != 0 && n != 0 {
					Zunt01('C', &m, &nsi, u, ldu, work, lwork, rwork, result.GetPtr(30))
					Zunt01('R', &nsi, &n, vt, ldvt, work, lwork, rwork, result.GetPtr(31))
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
				golapack.Zlacpy('F', &m, &n, asav, lda, a, lda)
				*srnamt = "ZGESVDX"
				golapack.Zgesvdx('V', 'V', 'V', &m, &n, a, lda, &vl, &vu, &il, &iu, &nsv, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZDRVBD: %s returned INFO=%6d.\n         M=%6d, N=%6d, JTYPE=%6d, LSWORK=%6d\n         ISEED=%5d\n", "GESVDX", iinfo, m, n, jtype, lswork, ioldsd)
					(*info) = abs(iinfo)
					return
				}
				//
				result.Set(32, zero)
				result.Set(33, zero)
				result.Set(34, zero)
				Zbdt05(&m, &n, asav, lda, s, &nsv, u.CVector(0, 0), ldu, vt, ldvt, work, result.GetPtr(32))
				if m != 0 && n != 0 {
					Zunt01('C', &m, &nsv, u, ldu, work, lwork, rwork, result.GetPtr(33))
					Zunt01('R', &nsv, &n, vt, ldvt, work, lwork, rwork, result.GetPtr(34))
				}

				//              End of Loop -- Check for RESULT(j) > THRESH
				ntest = 0
				nfail = 0
				for j = 1; j <= 39; j++ {
					if result.Get(j-1) >= zero {
						ntest = ntest + 1
					}
					if result.Get(j-1) >= (*thresh) {
						nfail = nfail + 1
					}
				}

				if nfail > 0 {
					ntestf = ntestf + 1
				}
				if ntestf == 1 {
					fmt.Printf(" SVD -- Complex Singular Value Decomposition Driver \n Matrix types (see ZDRVBD for details):\n\n 1 = Zero matrix\n 2 = Identity matrix\n 3 = Evenly spaced singular values near 1\n 4 = Evenly spaced singular values near underflow\n 5 = Evenly spaced singular values near overflow\n\n Tests performed: ( A is dense, U and V are unitary,\n                    S is an array, and Upartial, VTpartial, and\n                    Spartial are partially computed U, VT and S),\n\n")
					fmt.Printf(" Tests performed with Test Threshold = %8.2f\n ZGESVD: \n 1 = | A - U diag(S) VT | / ( |A| max(M,N) ulp ) \n 2 = | I - U**T U | / ( M ulp ) \n 3 = | I - VT VT**T | / ( N ulp ) \n 4 = 0 if S contains min(M,N) nonnegative values in decreasing order, else 1/ulp\n 5 = | U - Upartial | / ( M ulp )\n 6 = | VT - VTpartial | / ( N ulp )\n 7 = | S - Spartial | / ( min(M,N) ulp |S| )\n ZGESDD: \n 8 = | A - U diag(S) VT | / ( |A| max(M,N) ulp ) \n 9 = | I - U**T U | / ( M ulp ) \n10 = | I - VT VT**T | / ( N ulp ) \n11 = 0 if S contains min(M,N) nonnegative values in decreasing order, else 1/ulp\n12 = | U - Upartial | / ( M ulp )\n13 = | VT - VTpartial | / ( N ulp )\n14 = | S - Spartial | / ( min(M,N) ulp |S| )\n ZGESVJ: \n\n15 = | A - U diag(S) VT | / ( |A| max(M,N) ulp ) \n16 = | I - U**T U | / ( M ulp ) \n17 = | I - VT VT**T | / ( N ulp ) \n18 = 0 if S contains min(M,N) nonnegative values in decreasing order, else 1/ulp\n ZGESJV: \n\n19 = | A - U diag(S) VT | / ( |A| max(M,N) ulp )\n20 = | I - U**T U | / ( M ulp ) \n21 = | I - VT VT**T | / ( N ulp ) \n22 = 0 if S contains min(M,N) nonnegative values in decreasing order, else 1/ulp\n ZGESVDX(V,V,A): \n23 = | A - U diag(S) VT | / ( |A| max(M,N) ulp ) \n24 = | I - U**T U | / ( M ulp ) \n25 = | I - VT VT**T | / ( N ulp ) \n26 = 0 if S contains min(M,N) nonnegative values in decreasing order, else 1/ulp\n27 = | U - Upartial | / ( M ulp )\n28 = | VT - VTpartial | / ( N ulp )\n29 = | S - Spartial | / ( min(M,N) ulp |S| )\n ZGESVDX(V,V,I): \n30 = | U**T A VT**T - diag(S) | / ( |A| max(M,N) ulp )\n31 = | I - U**T U | / ( M ulp ) \n32 = | I - VT VT**T | / ( N ulp ) \n ZGESVDX(V,V,V) \n33 = | U**T A VT**T - diag(S) | / ( |A| max(M,N) ulp )\n34 = | I - U**T U | / ( M ulp ) \n35 = | I - VT VT**T | / ( N ulp )  ZGESVDQ(H,N,N,A,A\n36 = | A - U diag(S) VT | / ( |A| max(M,N) ulp ) \n37 = | I - U**T U | / ( M ulp ) \n38 = | I - VT VT**T | / ( N ulp ) \n39 = 0 if S contains min(M,N) nonnegative values in decreasing order, else 1/ulp\n\n\n", *thresh)
					ntestf = 2
				}

				for j = 1; j <= 39; j++ {
					if result.Get(j-1) >= (*thresh) {
						t.Fail()
						fmt.Printf(" M=%5d, N=%5d, _type %1d, IWS=%1d, seed=%4d, test(%2d)=%11.4f\n", m, n, jtype, iwspc, ioldsd, j, result.Get(j-1))
					}
				}

				nerrs = nerrs + nfail
				ntestt = ntestt + ntest

			}

		label220:
		}
	}

	//     Summary
	Alasvm([]byte("ZBD"), &nerrs, &ntestt, func() *int { y := 0; return &y }())
}
