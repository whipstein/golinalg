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

// Ddrvbd checks the singular value decomposition (SVD) drivers
// DGESVD, DGESDD, DGESVDQ, DGESVJ, DGEJSV, and DGESVDX.
//
// Both DGESVD and DGESDD factor A = U diag(S) VT, where U and VT are
// orthogonal and diag(S) is diagonal with the entries of the array S
// on its diagonal. The entries of S are the singular values,
// nonnegative and stored in decreasing order.  U and VT can be
// optionally not computed, overwritten on A, or computed partially.
//
// A is M by N. Let MNMIN = minint( M, N ). S has dimension MNMIN.
// U can be M by M or M by MNMIN. VT can be N by N or MNMIN by N.
//
// When DDRVBD is called, a number of matrix "sizes" (M's and N's)
// and a number of matrix "types" are specified.  For each size (M,N)
// and each type of matrix, and for the minimal workspace as well as
// workspace adequate to permit blocking, an  M x N  matrix "A" will be
// generated and used to test the SVD routines.  For each matrix, A will
// be factored as A = U diag(S) VT and the following 12 tests computed:
//
// Test for DGESVD:
//
// (1)    | A - U diag(S) VT | / ( |A| maxint(M,N) ulp )
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
// Test for DGESDD:
//
// (8)    | A - U diag(S) VT | / ( |A| maxint(M,N) ulp )
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
// Test for DGESVDQ:
//
// (36)   | A - U diag(S) VT | / ( |A| maxint(M,N) ulp )
//
// (37)   | I - U'U | / ( M ulp )
//
// (38)   | I - VT VT' | / ( N ulp )
//
// (39)   S contains MNMIN nonnegative values in decreasing order.
//        (Return 0 if true, 1/ULP if false.)
//
// Test for DGESVJ:
//
// (15)   | A - U diag(S) VT | / ( |A| maxint(M,N) ulp )
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
// (19)   | A - U diag(S) VT | / ( |A| maxint(M,N) ulp )
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
// (23)   | A - U diag(S) VT | / ( |A| maxint(M,N) ulp )
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
// (30)   | U' A VT''' - diag(S) | / ( |A| maxint(M,N) ulp )
//
// (31)   | I - U'U | / ( M ulp )
//
// (32)   | I - VT VT' | / ( N ulp )
//
// Test for DGESVDX( 'V', 'V', 'V' )
//
// (33)   | U' A VT''' - diag(S) | / ( |A| maxint(M,N) ulp )
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
func Ddrvbd(nsizes *int, mm *[]int, nn *[]int, ntypes *int, dotype *[]bool, iseed *[]int, thresh *float64, a *mat.Matrix, lda *int, u *mat.Matrix, ldu *int, vt *mat.Matrix, ldvt *int, asav, usav, vtsav *mat.Matrix, s, ssav, e, work *mat.Vector, lwork *int, iwork *[]int, nout, info *int, t *testing.T) {
	var badmm, badnn bool
	var jobq, jobu, jobvt, _range byte
	var anorm, dif, div, half, one, ovfl, rtunfl, two, ulp, ulpinv, unfl, vl, vu, zero float64
	var i, iinfo, ijq, iju, ijvt, il, itemp, iu, iws, iwtmp, j, jsize, jtype, liwork, lrwork, lswork, m, maxtyp, minwrk, mmax, mnmax, mnmin, mtypes, n, nfail, nmax, ns, nsi, nsv, ntest, numrank int

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
	*info = 0
	badmm = false
	badnn = false
	mmax = 1
	nmax = 1
	mnmax = 1
	minwrk = 1
	for j = 1; j <= (*nsizes); j++ {
		mmax = maxint(mmax, (*mm)[j-1])
		if (*mm)[j-1] < 0 {
			badmm = true
		}
		nmax = maxint(nmax, (*nn)[j-1])
		if (*nn)[j-1] < 0 {
			badnn = true
		}
		mnmax = maxint(mnmax, minint((*mm)[j-1], (*nn)[j-1]))
		minwrk = maxint(minwrk, maxint(3*minint((*mm)[j-1], (*nn)[j-1])+maxint((*mm)[j-1], (*nn)[j-1]), 5*minint((*mm)[j-1], (*nn)[j-1]-4))+2*int(math.Pow(float64(minint((*mm)[j-1], (*nn)[j-1])), 2)))
	}

	//     Check for errors
	if (*nsizes) < 0 {
		*info = -1
	} else if badmm {
		*info = -2
	} else if badnn {
		*info = -3
	} else if (*ntypes) < 0 {
		*info = -4
	} else if (*lda) < maxint(1, mmax) {
		*info = -10
	} else if (*ldu) < maxint(1, mmax) {
		*info = -12
	} else if (*ldvt) < maxint(1, nmax) {
		*info = -14
	} else if minwrk > (*lwork) {
		*info = -21
	}

	if *info != 0 {
		gltest.Xerbla([]byte("DDRVBD"), -*info)
		return
	}

	//     Initialize constants
	path := []byte("DBD")
	nfail = 0
	ntest = 0
	unfl = golapack.Dlamch(SafeMinimum)
	ovfl = one / unfl
	golapack.Dlabad(&unfl, &ovfl)
	ulp = golapack.Dlamch(Precision)
	rtunfl = math.Sqrt(unfl)
	ulpinv = one / ulp
	(*infot) = 0

	//     Loop over sizes, types
	for jsize = 1; jsize <= (*nsizes); jsize++ {
		m = (*mm)[jsize-1]
		n = (*nn)[jsize-1]
		mnmin = minint(m, n)

		if (*nsizes) != 1 {
			mtypes = minint(maxtyp, *ntypes)
		} else {
			mtypes = minint(maxtyp+1, *ntypes)
		}

		for jtype = 1; jtype <= mtypes; jtype++ {
			if !(*dotype)[jtype-1] {
				goto label230
			}

			for j = 1; j <= 4; j++ {
				ioldsd[j-1] = (*iseed)[j-1]
			}

			//           Compute "A"
			if mtypes > maxtyp {
				goto label30
			}

			if jtype == 1 {
				//              Zero matrix
				golapack.Dlaset('F', &m, &n, &zero, &zero, a, lda)

			} else if jtype == 2 {
				//              Identity matrix
				golapack.Dlaset('F', &m, &n, &zero, &one, a, lda)

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
				matgen.Dlatms(&m, &n, 'U', iseed, 'N', s, toPtr(4), toPtrf64(float64(mnmin)), &anorm, toPtr(m-1), toPtr(n-1), 'N', a, lda, work, &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" DDRVBD: %s returned INFO=%6d.\n         M=%6d, N=%6d, JTYPE=%6d, ISEED=%5d\n", "Generator", iinfo, m, n, jtype, ioldsd)
					*info = absint(iinfo)
					return
				}
			}

		label30:
			;
			golapack.Dlacpy('F', &m, &n, a, lda, asav, lda)

			//           Do for minimal and adequate (for blocking) workspace
			for iws = 1; iws <= 4; iws++ {

				for j = 1; j <= 32; j++ {
					result.Set(j-1, -one)
				}

				//              Test DGESVD: Factorize A
				iwtmp = maxint(3*minint(m, n)+maxint(m, n), 5*minint(m, n))
				lswork = iwtmp + (iws-1)*((*lwork)-iwtmp)/3
				lswork = minint(lswork, *lwork)
				lswork = maxint(lswork, 1)
				if iws == 4 {
					lswork = (*lwork)
				}

				if iws > 1 {
					golapack.Dlacpy('F', &m, &n, asav, lda, a, lda)
				}
				*srnamt = "DGESVD"
				golapack.Dgesvd('A', 'A', &m, &n, a, lda, ssav, usav, ldu, vtsav, ldvt, work, &lswork, &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" DDRVBD: %s returned INFO=%6d.\n         M=%6d, N=%6d, JTYPE=%6d, LSWORK=%6d\n         ISEED=%5d\n", "GESVD", iinfo, m, n, jtype, lswork, ioldsd)
					*info = absint(iinfo)
					return
				}

				//              Do tests 1--4
				Dbdt01(&m, &n, toPtr(0), asav, lda, usav, ldu, ssav, e, vtsav, ldvt, work, result.GetPtr(0))
				if m != 0 && n != 0 {
					Dort01('C', &m, &m, usav, ldu, work, lwork, result.GetPtr(1))
					Dort01('R', &n, &n, vtsav, ldvt, work, lwork, result.GetPtr(2))
				}
				result.Set(3, zero)
				for i = 1; i <= mnmin-1; i++ {
					if ssav.Get(i-1) < ssav.Get(i+1-1) {
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
						jobu = cjob[iju+1-1]
						jobvt = cjob[ijvt+1-1]
						golapack.Dlacpy('F', &m, &n, asav, lda, a, lda)
						*srnamt = "DGESVD"
						golapack.Dgesvd(jobu, jobvt, &m, &n, a, lda, s, u, ldu, vt, ldvt, work, &lswork, &iinfo)

						//                    Compare U
						dif = zero
						if m > 0 && n > 0 {
							if iju == 1 {
								Dort03('C', &m, &mnmin, &m, &mnmin, usav, ldu, a, lda, work, lwork, &dif, &iinfo)
							} else if iju == 2 {
								Dort03('C', &m, &mnmin, &m, &mnmin, usav, ldu, u, ldu, work, lwork, &dif, &iinfo)
							} else if iju == 3 {
								Dort03('C', &m, &m, &m, &mnmin, usav, ldu, u, ldu, work, lwork, &dif, &iinfo)
							}
						}
						result.Set(4, maxf64(result.Get(4), dif))

						//                    Compare VT
						dif = zero
						if m > 0 && n > 0 {
							if ijvt == 1 {
								Dort03('R', &n, &mnmin, &n, &mnmin, vtsav, ldvt, a, lda, work, lwork, &dif, &iinfo)
							} else if ijvt == 2 {
								Dort03('R', &n, &mnmin, &n, &mnmin, vtsav, ldvt, vt, ldvt, work, lwork, &dif, &iinfo)
							} else if ijvt == 3 {
								Dort03('R', &n, &n, &n, &mnmin, vtsav, ldvt, vt, ldvt, work, lwork, &dif, &iinfo)
							}
						}
						result.Set(5, maxf64(result.Get(5), dif))

						//                    Compare S
						dif = zero
						div = maxf64(float64(mnmin)*ulp*s.Get(0), unfl)
						for i = 1; i <= mnmin-1; i++ {
							if ssav.Get(i-1) < ssav.Get(i+1-1) {
								dif = ulpinv
							}
							if ssav.Get(i-1) < zero {
								dif = ulpinv
							}
							dif = maxf64(dif, math.Abs(ssav.Get(i-1)-s.Get(i-1))/div)
						}
						result.Set(6, maxf64(result.Get(6), dif))
					label70:
					}
				}

				//              Test DGESDD: Factorize A
				iwtmp = 5*mnmin*mnmin + 9*mnmin + maxint(m, n)
				lswork = iwtmp + (iws-1)*((*lwork)-iwtmp)/3
				lswork = minint(lswork, *lwork)
				lswork = maxint(lswork, 1)
				if iws == 4 {
					lswork = (*lwork)
				}

				golapack.Dlacpy('F', &m, &n, asav, lda, a, lda)
				*srnamt = "DGESDD"
				golapack.Dgesdd('A', &m, &n, a, lda, ssav, usav, ldu, vtsav, ldvt, work, &lswork, iwork, &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" DDRVBD: %s returned INFO=%6d.\n         M=%6d, N=%6d, JTYPE=%6d, LSWORK=%6d\n         ISEED=%5d\n", "GESDD", iinfo, m, n, jtype, lswork, ioldsd)
					*info = absint(iinfo)
					return
				}

				//              Do tests 8--11
				Dbdt01(&m, &n, toPtr(0), asav, lda, usav, ldu, ssav, e, vtsav, ldvt, work, result.GetPtr(7))
				if m != 0 && n != 0 {
					Dort01('C', &m, &m, usav, ldu, work, lwork, result.GetPtr(8))
					Dort01('R', &n, &n, vtsav, ldvt, work, lwork, result.GetPtr(9))
				}
				result.Set(10, zero)
				for i = 1; i <= mnmin-1; i++ {
					if ssav.Get(i-1) < ssav.Get(i+1-1) {
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
					jobq = cjob[ijq+1-1]
					golapack.Dlacpy('F', &m, &n, asav, lda, a, lda)
					*srnamt = "DGESDD"
					golapack.Dgesdd(jobq, &m, &n, a, lda, s, u, ldu, vt, ldvt, work, &lswork, iwork, &iinfo)

					//                 Compare U
					dif = zero
					if m > 0 && n > 0 {
						if ijq == 1 {
							if m >= n {
								Dort03('C', &m, &mnmin, &m, &mnmin, usav, ldu, a, lda, work, lwork, &dif, info)
							} else {
								Dort03('C', &m, &mnmin, &m, &mnmin, usav, ldu, u, ldu, work, lwork, &dif, info)
							}
						} else if ijq == 2 {
							Dort03('C', &m, &mnmin, &m, &mnmin, usav, ldu, u, ldu, work, lwork, &dif, info)
						}
					}
					result.Set(11, maxf64(result.Get(11), dif))

					//                 Compare VT
					dif = zero
					if m > 0 && n > 0 {
						if ijq == 1 {
							if m >= n {
								Dort03('R', &n, &mnmin, &n, &mnmin, vtsav, ldvt, vt, ldvt, work, lwork, &dif, info)
							} else {
								Dort03('R', &n, &mnmin, &n, &mnmin, vtsav, ldvt, a, lda, work, lwork, &dif, info)
							}
						} else if ijq == 2 {
							Dort03('R', &n, &mnmin, &n, &mnmin, vtsav, ldvt, vt, ldvt, work, lwork, &dif, info)
						}
					}
					result.Set(12, maxf64(result.Get(12), dif))

					//                 Compare S
					dif = zero
					div = maxf64(float64(mnmin)*ulp*s.Get(0), unfl)
					for i = 1; i <= mnmin-1; i++ {
						if ssav.Get(i-1) < ssav.Get(i+1-1) {
							dif = ulpinv
						}
						if ssav.Get(i-1) < zero {
							dif = ulpinv
						}
						dif = maxf64(dif, math.Abs(ssav.Get(i-1)-s.Get(i-1))/div)
					}
					result.Set(13, maxf64(result.Get(13), dif))
				}

				//              Test DGESVDQ
				//              Note: DGESVDQ only works for M >= N
				result.Set(35, zero)
				result.Set(36, zero)
				result.Set(37, zero)
				result.Set(38, zero)

				if m >= n {
					iwtmp = 5*mnmin*mnmin + 9*mnmin + maxint(m, n)
					lswork = iwtmp + (iws-1)*((*lwork)-iwtmp)/3
					lswork = minint(lswork, *lwork)
					lswork = maxint(lswork, 1)
					if iws == 4 {
						lswork = (*lwork)
					}

					golapack.Dlacpy('F', &m, &n, asav, lda, a, lda)
					*srnamt = "DGESVDQ"

					lrwork = 2
					liwork = maxint(n, 1)
					golapack.Dgesvdq('H', 'N', 'N', 'A', 'A', &m, &n, a, lda, ssav, usav, ldu, vtsav, ldvt, &numrank, iwork, &liwork, work, lwork, rwork, &lrwork, &iinfo)

					if iinfo != 0 {
						t.Fail()
						fmt.Printf(" DDRVBD: %s returned INFO=%6d.\n         M=%6d, N=%6d, JTYPE=%6d, LSWORK=%6d\n         ISEED=%5d\n", "DGESVDQ", iinfo, m, n, jtype, lswork, ioldsd)
						*info = absint(iinfo)
						return
					}

					//                 Do tests 36--39
					Dbdt01(&m, &n, toPtr(0), asav, lda, usav, ldu, ssav, e, vtsav, ldvt, work, result.GetPtr(35))
					if m != 0 && n != 0 {
						Dort01('C', &m, &m, usav, ldu, work, lwork, result.GetPtr(36))
						Dort01('R', &n, &n, vtsav, ldvt, work, lwork, result.GetPtr(37))
					}
					result.Set(38, zero)
					for i = 1; i <= mnmin-1; i++ {
						if ssav.Get(i-1) < ssav.Get(i+1-1) {
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

				//              Test DGESVJ
				//              Note: DGESVJ only works for M >= N
				result.Set(14, zero)
				result.Set(15, zero)
				result.Set(16, zero)
				result.Set(17, zero)

				if m >= n {
					iwtmp = 5*mnmin*mnmin + 9*mnmin + maxint(m, n)
					lswork = iwtmp + (iws-1)*((*lwork)-iwtmp)/3
					lswork = minint(lswork, *lwork)
					lswork = maxint(lswork, 1)
					if iws == 4 {
						lswork = (*lwork)
					}

					golapack.Dlacpy('F', &m, &n, asav, lda, usav, lda)
					*srnamt = "DGESVJ"
					golapack.Dgesvj('G', 'U', 'V', &m, &n, usav, lda, ssav, toPtr(0), a, ldvt, work, lwork, info)

					//                 DGESVJ returns V not VT
					for j = 1; j <= n; j++ {
						for i = 1; i <= n; i++ {
							vtsav.Set(j-1, i-1, a.Get(i-1, j-1))
						}
					}

					if iinfo != 0 {
						t.Fail()
						fmt.Printf(" DDRVBD: %s returned INFO=%6d.\n         M=%6d, N=%6d, JTYPE=%6d, LSWORK=%6d\n         ISEED=%5d\n", "GESVJ", iinfo, m, n, jtype, lswork, ioldsd)
						*info = absint(iinfo)
						return
					}

					//                 Do tests 15--18
					Dbdt01(&m, &n, toPtr(0), asav, lda, usav, ldu, ssav, e, vtsav, ldvt, work, result.GetPtr(14))
					if m != 0 && n != 0 {
						Dort01('C', &m, &m, usav, ldu, work, lwork, result.GetPtr(15))
						Dort01('R', &n, &n, vtsav, ldvt, work, lwork, result.GetPtr(16))
					}
					result.Set(17, zero)
					for i = 1; i <= mnmin-1; i++ {
						if ssav.Get(i-1) < ssav.Get(i+1-1) {
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
					iwtmp = 5*mnmin*mnmin + 9*mnmin + maxint(m, n)
					lswork = iwtmp + (iws-1)*((*lwork)-iwtmp)/3
					lswork = minint(lswork, *lwork)
					lswork = maxint(lswork, 1)
					if iws == 4 {
						lswork = (*lwork)
					}

					golapack.Dlacpy('F', &m, &n, asav, lda, vtsav, lda)
					*srnamt = "DGEJSV"
					golapack.Dgejsv('G', 'U', 'V', 'R', 'N', 'N', &m, &n, vtsav, lda, ssav, usav, ldu, a, ldvt, work, lwork, iwork, info)

					//                 DGEJSV returns V not VT
					for j = 1; j <= n; j++ {
						for i = 1; i <= n; i++ {
							vtsav.Set(j-1, i-1, a.Get(i-1, j-1))
						}
					}

					if iinfo != 0 {
						t.Fail()
						fmt.Printf(" DDRVBD: %s returned INFO=%6d.\n         M=%6d, N=%6d, JTYPE=%6d, LSWORK=%6d\n         ISEED=%5d\n", "GEJSV", iinfo, m, n, jtype, lswork, ioldsd)
						*info = absint(iinfo)
						return
					}

					//                 Do tests 19--22
					Dbdt01(&m, &n, toPtr(0), asav, lda, usav, ldu, ssav, e, vtsav, ldvt, work, result.GetPtr(18))
					if m != 0 && n != 0 {
						Dort01('C', &m, &m, usav, ldu, work, lwork, result.GetPtr(19))
						Dort01('R', &n, &n, vtsav, ldvt, work, lwork, result.GetPtr(20))
					}
					result.Set(21, zero)
					for i = 1; i <= mnmin-1; i++ {
						if ssav.Get(i-1) < ssav.Get(i+1-1) {
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
				golapack.Dlacpy('F', &m, &n, asav, lda, a, lda)
				golapack.Dgesvdx('V', 'V', 'A', &m, &n, a, lda, &vl, &vu, &il, &iu, &ns, ssav, usav, ldu, vtsav, ldvt, work, lwork, iwork, &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" DDRVBD: %s returned INFO=%6d.\n         M=%6d, N=%6d, JTYPE=%6d, LSWORK=%6d\n         ISEED=%5d\n", "GESVDX", iinfo, m, n, jtype, lswork, ioldsd)
					*info = absint(iinfo)
					return
				}

				//              Do tests 23--29
				result.Set(22, zero)
				result.Set(23, zero)
				result.Set(24, zero)
				Dbdt01(&m, &n, toPtr(0), asav, lda, usav, ldu, ssav, e, vtsav, ldvt, work, result.GetPtr(22))
				if m != 0 && n != 0 {
					Dort01('C', &m, &m, usav, ldu, work, lwork, result.GetPtr(23))
					Dort01('R', &n, &n, vtsav, ldvt, work, lwork, result.GetPtr(24))
				}
				result.Set(25, zero)
				for i = 1; i <= mnmin-1; i++ {
					if ssav.Get(i-1) < ssav.Get(i+1-1) {
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
						jobu = cjobv[iju+1-1]
						jobvt = cjobv[ijvt+1-1]
						_range = cjobr[0]
						golapack.Dlacpy('F', &m, &n, asav, lda, a, lda)
						golapack.Dgesvdx(jobu, jobvt, _range, &m, &n, a, lda, &vl, &vu, &il, &iu, &ns, s, u, ldu, vt, ldvt, work, lwork, iwork, &iinfo)

						//                    Compare U
						dif = zero
						if m > 0 && n > 0 {
							if iju == 1 {
								Dort03('C', &m, &mnmin, &m, &mnmin, usav, ldu, u, ldu, work, lwork, &dif, &iinfo)
							}
						}
						result.Set(26, maxf64(result.Get(26), dif))

						//                    Compare VT
						dif = zero
						if m > 0 && n > 0 {
							if ijvt == 1 {
								Dort03('R', &n, &mnmin, &n, &mnmin, vtsav, ldvt, vt, ldvt, work, lwork, &dif, &iinfo)
							}
						}
						result.Set(27, maxf64(result.Get(27), dif))

						//                    Compare S
						dif = zero
						div = maxf64(float64(mnmin)*ulp*s.Get(0), unfl)
						for i = 1; i <= mnmin-1; i++ {
							if ssav.Get(i-1) < ssav.Get(i+1-1) {
								dif = ulpinv
							}
							if ssav.Get(i-1) < zero {
								dif = ulpinv
							}
							dif = maxf64(dif, math.Abs(ssav.Get(i-1)-s.Get(i-1))/div)
						}
						result.Set(28, maxf64(result.Get(28), dif))
					label170:
					}
				}

				//              Do tests 30--32: DGESVDX( 'V', 'V', 'I' )
				for i = 1; i <= 4; i++ {
					iseed2[i-1] = (*iseed)[i-1]
				}
				if mnmin <= 1 {
					il = 1
					iu = maxint(1, mnmin)
				} else {
					il = 1 + int(float64(mnmin-1)*matgen.Dlarnd(toPtr(1), &iseed2))
					iu = 1 + int(float64(mnmin-1)*matgen.Dlarnd(toPtr(1), &iseed2))
					if iu < il {
						itemp = iu
						iu = il
						il = itemp
					}
				}
				golapack.Dlacpy('F', &m, &n, asav, lda, a, lda)
				golapack.Dgesvdx('V', 'V', 'I', &m, &n, a, lda, &vl, &vu, &il, &iu, &nsi, s, u, ldu, vt, ldvt, work, lwork, iwork, &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" DDRVBD: %s returned INFO=%6d.\n         M=%6d, N=%6d, JTYPE=%6d, LSWORK=%6d\n         ISEED=%5d\n", "GESVDX", iinfo, m, n, jtype, lswork, ioldsd)
					*info = absint(iinfo)
					return
				}

				result.Set(29, zero)
				result.Set(30, zero)
				result.Set(31, zero)
				Dbdt05(&m, &n, asav, lda, s, &nsi, u, ldu, vt, ldvt, work, result.GetPtr(29))
				Dort01('C', &m, &nsi, u, ldu, work, lwork, result.GetPtr(30))
				Dort01('R', &nsi, &n, vt, ldvt, work, lwork, result.GetPtr(31))

				//              Do tests 33--35: DGESVDX( 'V', 'V', 'V' )
				if mnmin > 0 && nsi > 1 {
					if il != 1 {
						vu = ssav.Get(il-1) + maxf64(half*math.Abs(ssav.Get(il-1)-ssav.Get(il-1-1)), ulp*anorm, two*rtunfl)
					} else {
						vu = ssav.Get(0) + maxf64(half*math.Abs(ssav.Get(ns-1)-ssav.Get(0)), ulp*anorm, two*rtunfl)
					}
					if iu != ns {
						vl = ssav.Get(iu-1) - maxf64(ulp*anorm, two*rtunfl, half*math.Abs(ssav.Get(iu+1-1)-ssav.Get(iu-1)))
					} else {
						vl = ssav.Get(ns-1) - maxf64(ulp*anorm, two*rtunfl, half*math.Abs(ssav.Get(ns-1)-ssav.Get(0)))
					}
					vl = maxf64(vl, zero)
					vu = maxf64(vu, zero)
					if vl >= vu {
						vu = maxf64(vu*2, vu+vl+half)
					}
				} else {
					vl = zero
					vu = one
				}
				golapack.Dlacpy('F', &m, &n, asav, lda, a, lda)
				golapack.Dgesvdx('V', 'V', 'V', &m, &n, a, lda, &vl, &vu, &il, &iu, &nsv, s, u, ldu, vt, ldvt, work, lwork, iwork, &iinfo)
				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" DDRVBD: %s returned INFO=%6d.\n         M=%6d, N=%6d, JTYPE=%6d, LSWORK=%6d\n         ISEED=%5d\n", "GESVDX", iinfo, m, n, jtype, lswork, ioldsd)
					*info = absint(iinfo)
					return
				}

				result.Set(32, zero)
				result.Set(33, zero)
				result.Set(34, zero)
				Dbdt05(&m, &n, asav, lda, s, &nsv, u, ldu, vt, ldvt, work, result.GetPtr(32))
				Dort01('C', &m, &nsv, u, ldu, work, lwork, result.GetPtr(33))
				Dort01('R', &nsv, &n, vt, ldvt, work, lwork, result.GetPtr(34))

				//              End of Loop -- Check for RESULT(j) > THRESH
				for j = 1; j <= 39; j++ {
					if result.Get(j-1) >= (*thresh) {
						t.Fail()
						if nfail == 0 {
							fmt.Printf(" SVD -- Real Singular Value Decomposition Driver \n Matrix types (see DDRVBD for details):\n\n 1 = Zero matrix\n 2 = Identity matrix\n 3 = Evenly spaced singular values near 1\n 4 = Evenly spaced singular values near underflow\n 5 = Evenly spaced singular values near overflow\n\n Tests performed: ( A is dense, U and V are orthogonal,\n                    S is an array, and Upartial, VTpartial, and\n                    Spartial are partially computed U, VT and S),\n\n")
							fmt.Printf(" 1 = | A - U diag(S) VT | / ( |A| maxint(M,N) ulp ) \n 2 = | I - U**T U | / ( M ulp ) \n 3 = | I - VT VT**T | / ( N ulp ) \n 4 = 0 if S contains minint(M,N) nonnegative values in decreasing order, else 1/ulp\n 5 = | U - Upartial | / ( M ulp )\n 6 = | VT - VTpartial | / ( N ulp )\n 7 = | S - Spartial | / ( minint(M,N) ulp |S| )\n 8 = | A - U diag(S) VT | / ( |A| maxint(M,N) ulp ) \n 9 = | I - U**T U | / ( M ulp ) \n10 = | I - VT VT**T | / ( N ulp ) \n11 = 0 if S contains minint(M,N) nonnegative values in decreasing order, else 1/ulp\n12 = | U - Upartial | / ( M ulp )\n13 = | VT - VTpartial | / ( N ulp )\n14 = | S - Spartial | / ( minint(M,N) ulp |S| )\n15 = | A - U diag(S) VT | / ( |A| maxint(M,N) ulp ) \n16 = | I - U**T U | / ( M ulp ) \n17 = | I - VT VT**T | / ( N ulp ) \n18 = 0 if S contains minint(M,N) nonnegative values in decreasing order, else 1/ulp\n19 = | U - Upartial | / ( M ulp )\n20 = | VT - VTpartial | / ( N ulp )\n21 = | S - Spartial | / ( minint(M,N) ulp |S| )\n22 = 0 if S contains minint(M,N) nonnegative values in decreasing order, else 1/ulp\n23 = | A - U diag(S) VT | / ( |A| maxint(M,N) ulp ), DGESVDX(V,V,A) \n24 = | I - U**T U | / ( M ulp ) \n25 = | I - VT VT**T | / ( N ulp ) \n26 = 0 if S contains minint(M,N) nonnegative values in decreasing order, else 1/ulp\n27 = | U - Upartial | / ( M ulp )\n28 = | VT - VTpartial | / ( N ulp )\n29 = | S - Spartial | / ( minint(M,N) ulp |S| )\n30 = | U**T A VT**T - diag(S) | / ( |A| maxint(M,N) ulp ), DGESVDX(V,V,I) \n31 = | I - U**T U | / ( M ulp ) \n32 = | I - VT VT**T | / ( N ulp ) \n33 = | U**T A VT**T - diag(S) | / ( |A| maxint(M,N) ulp ), DGESVDX(V,V,V) \n34 = | I - U**T U | / ( M ulp ) \n35 = | I - VT VT**T | / ( N ulp )  DGESVDQ(H,N,N,A,A\n36 = | A - U diag(S) VT | / ( |A| maxint(M,N) ulp ) \n37 = | I - U**T U | / ( M ulp ) \n38 = | I - VT VT**T | / ( N ulp ) \n39 = 0 if S contains minint(M,N) nonnegative values in decreasing order, else 1/ulp\n\n\n")
						}
						fmt.Printf(" M=%5d, N=%5d, type %1d, IWS=%1d, seed=%4d, test(%2d)=%11.4f\n", m, n, jtype, iws, ioldsd, j, result.Get(j-1))
						nfail = nfail + 1
					}
				}
				ntest = ntest + 39
			}
		label230:
		}
	}

	//     Summary
	Alasvm(path, &nfail, &ntest, toPtr(0))
}
