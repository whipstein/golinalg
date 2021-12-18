package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dggevx computes for a pair of N-by-N real nonsymmetric matrices (A,B)
// the generalized eigenvalues, and optionally, the left and/or right
// generalized eigenvectors.
//
// Optionally also, it computes a balancing transformation to improve
// the conditioning of the eigenvalues and eigenvectors (ILO, IHI,
// LSCALE, RSCALE, ABNRM, and BBNRM), reciprocal condition numbers for
// the eigenvalues (RCONDE), and reciprocal condition numbers for the
// right eigenvectors (RCONDV).
//
// A generalized eigenvalue for a pair of matrices (A,B) is a scalar
// lambda or a ratio alpha/beta = lambda, such that A - lambda*B is
// singular. It is usually represented as the pair (alpha,beta), as
// there is a reasonable interpretation for beta=0, and even for both
// being zero.
//
// The right eigenvector v(j) corresponding to the eigenvalue lambda(j)
// of (A,B) satisfies
//
//                  A * v(j) = lambda(j) * B * v(j) .
//
// The left eigenvector u(j) corresponding to the eigenvalue lambda(j)
// of (A,B) satisfies
//
//                  u(j)**H * A  = lambda(j) * u(j)**H * B.
//
// where u(j)**H is the conjugate-transpose of u(j).
func Dggevx(balanc, jobvl, jobvr, sense byte, n int, a, b *mat.Matrix, alphar, alphai, beta *mat.Vector, vl, vr *mat.Matrix, lscale, rscale *mat.Vector, rconde, rcondv, work *mat.Vector, lwork int, iwork *[]int, bwork *[]bool) (ilo, ihi int, abnrm, bbnrm float64, info int, err error) {
	var ilascl, ilbscl, ilv, ilvl, ilvr, lquery, noscl, pair, wantsb, wantse, wantsn, wantsv bool
	var chtemp byte
	var anrm, anrmto, bignum, bnrm, bnrmto, eps, one, smlnum, temp, zero float64
	var i, icols, ierr, ijobvl, ijobvr, irows, itau, iwrk, iwrk1, j, jc, jr, maxwrk, minwrk, mm int

	ldumma := make([]bool, 1)

	zero = 0.0
	one = 1.0

	//     Decode the input arguments
	if jobvl == 'N' {
		ijobvl = 1
		ilvl = false
	} else if jobvl == 'V' {
		ijobvl = 2
		ilvl = true
	} else {
		ijobvl = -1
		ilvl = false
	}

	if jobvr == 'N' {
		ijobvr = 1
		ilvr = false
	} else if jobvr == 'V' {
		ijobvr = 2
		ilvr = true
	} else {
		ijobvr = -1
		ilvr = false
	}
	ilv = ilvl || ilvr

	noscl = balanc == 'N' || balanc == 'P'
	wantsn = sense == 'N'
	wantse = sense == 'E'
	wantsv = sense == 'V'
	wantsb = sense == 'B'

	//     Test the input arguments
	lquery = (lwork == -1)
	if !(balanc == 'N' || balanc == 'S' || balanc == 'P' || balanc == 'B') {
		err = fmt.Errorf("!(balanc == 'N' || balanc == 'S' || balanc == 'P' || balanc == 'B'): balanc='%c'", balanc)
	} else if ijobvl <= 0 {
		err = fmt.Errorf("ijobvl <= 0: jobvl='%c'", jobvl)
	} else if ijobvr <= 0 {
		err = fmt.Errorf("ijobvr <= 0: jobvr='%c'", jobvr)
	} else if !(wantsn || wantse || wantsb || wantsv) {
		err = fmt.Errorf("!(wantsn || wantse || wantsb || wantsv): sense='%c'", sense)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	} else if vl.Rows < 1 || (ilvl && vl.Rows < n) {
		err = fmt.Errorf("vl.Rows < 1 || (ilvl && vl.Rows < n): vl.Rows=%v, ilvl=%v, n=%v", vl.Rows, ilvl, n)
	} else if vr.Rows < 1 || (ilvr && vr.Rows < n) {
		err = fmt.Errorf("vr.Rows < 1 || (ilvr && vr.Rows < n): vr.Rows=%v, ilvr=%v, n=%v", vr.Rows, ilvr, n)
	}

	//     Compute workspace
	//      (Note: Comments in the code beginning "Workspace:" describe the
	//       minimal amount of workspace needed at that point in the code,
	//       as well as the preferred amount for good performance.
	//       NB refers to the optimal block size for the immediately
	//       following subroutine, as returned by ILAENV. The workspace is
	//       computed assuming ILO = 1 and IHI = N, the worst case.)
	if err == nil {
		if n == 0 {
			minwrk = 1
			maxwrk = 1
		} else {
			if noscl && !ilv {
				minwrk = 2 * n
			} else {
				minwrk = 6 * n
			}
			if wantse || wantsb {
				minwrk = 10 * n
			}
			if wantsv || wantsb {
				minwrk = max(minwrk, 2*n*(n+4)+16)
			}
			maxwrk = minwrk
			maxwrk = max(maxwrk, n+n*Ilaenv(1, "Dgeqrf", []byte{' '}, n, 1, n, 0))
			maxwrk = max(maxwrk, n+n*Ilaenv(1, "Dormqr", []byte{' '}, n, 1, n, 0))
			if ilvl {
				maxwrk = max(maxwrk, n+n*Ilaenv(1, "Dorgqr", []byte{' '}, n, 1, n, 0))
			}
		}
		work.Set(0, float64(maxwrk))

		if lwork < minwrk && !lquery {
			err = fmt.Errorf("lwork < minwrk && !lquery: lwork=%v, minwrk=%v, lquery=%v", lwork, minwrk, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Dggevx", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Get machine constants
	eps = Dlamch(Precision)
	smlnum = Dlamch(SafeMinimum)
	bignum = one / smlnum
	smlnum, bignum = Dlabad(smlnum, bignum)
	smlnum = math.Sqrt(smlnum) / eps
	bignum = one / smlnum

	//     Scale A if math.Max element outside range [SMLNUM,BIGNUM]
	anrm = Dlange('M', n, n, a, work)
	ilascl = false
	if anrm > zero && anrm < smlnum {
		anrmto = smlnum
		ilascl = true
	} else if anrm > bignum {
		anrmto = bignum
		ilascl = true
	}
	if ilascl {
		if err = Dlascl('G', 0, 0, anrm, anrmto, n, n, a); err != nil {
			panic(err)
		}
	}

	//     Scale B if math.Max element outside range [SMLNUM,BIGNUM]
	bnrm = Dlange('M', n, n, b, work)
	ilbscl = false
	if bnrm > zero && bnrm < smlnum {
		bnrmto = smlnum
		ilbscl = true
	} else if bnrm > bignum {
		bnrmto = bignum
		ilbscl = true
	}
	if ilbscl {
		if err = Dlascl('G', 0, 0, bnrm, bnrmto, n, n, b); err != nil {
			panic(err)
		}
	}

	//     Permute and/or balance the matrix pair (A,B)
	//     (Workspace: need 6*N if BALANC = 'S' or 'B', 1 otherwise)
	if ilo, ihi, err = Dggbal(balanc, n, a, b, lscale, rscale, work); err != nil {
		panic(err)
	}

	//     Compute ABNRM and BBNRM
	abnrm = Dlange('1', n, n, a, work.Off(0))
	if ilascl {
		work.Set(0, abnrm)
		if err = Dlascl('G', 0, 0, anrmto, anrm, 1, 1, work.Matrix(1, opts)); err != nil {
			panic(err)
		}
		abnrm = work.Get(0)
	}

	bbnrm = Dlange('1', n, n, b, work.Off(0))
	if ilbscl {
		work.Set(0, bbnrm)
		if err = Dlascl('G', 0, 0, bnrmto, bnrm, 1, 1, work.Matrix(1, opts)); err != nil {
			panic(err)
		}
		bbnrm = work.Get(0)
	}

	//     Reduce B to triangular form (QR decomposition of B)
	//     (Workspace: need N, prefer N*NB )
	irows = ihi + 1 - ilo
	if ilv || !wantsn {
		icols = n + 1 - ilo
	} else {
		icols = irows
	}
	itau = 1
	iwrk = itau + irows
	if err = Dgeqrf(irows, icols, b.Off(ilo-1, ilo-1), work.Off(itau-1), work.Off(iwrk-1), lwork+1-iwrk); err != nil {
		panic(err)
	}

	//     Apply the orthogonal transformation to A
	//     (Workspace: need N, prefer N*NB)
	if err = Dormqr(Left, Trans, irows, icols, irows, b.Off(ilo-1, ilo-1), work.Off(itau-1), a.Off(ilo-1, ilo-1), work.Off(iwrk-1), lwork+1-iwrk); err != nil {
		panic(err)
	}

	//     Initialize VL and/or VR
	//     (Workspace: need N, prefer N*NB)
	if ilvl {
		Dlaset(Full, n, n, zero, one, vl)
		if irows > 1 {
			Dlacpy(Lower, irows-1, irows-1, b.Off(ilo, ilo-1), vl.Off(ilo, ilo-1))
		}
		if err = Dorgqr(irows, irows, irows, vl.Off(ilo-1, ilo-1), work.Off(itau-1), work.Off(iwrk-1), lwork+1-iwrk); err != nil {
			panic(err)
		}
	}

	if ilvr {
		Dlaset(Full, n, n, zero, one, vr)
	}

	//     Reduce to generalized Hessenberg form
	//     (Workspace: none needed)
	if ilv || !wantsn {
		//        Eigenvectors requested -- work on whole matrix.
		if err = Dgghrd(jobvl, jobvr, n, ilo, ihi, a, b, vl, vr); err != nil {
			panic(err)
		}
	} else {
		if err = Dgghrd('N', 'N', irows, 1, irows, a.Off(ilo-1, ilo-1), b.Off(ilo-1, ilo-1), vl, vr); err != nil {
			panic(err)
		}
	}

	//     Perform QZ algorithm (Compute eigenvalues, and optionally, the
	//     Schur forms and Schur vectors)
	//     (Workspace: need N)
	if ilv || !wantsn {
		chtemp = 'S'
	} else {
		chtemp = 'E'
	}

	if ierr, err = Dhgeqz(chtemp, jobvl, jobvr, n, ilo, ihi, a, b, alphar, alphai, beta, vl, vr, work, lwork); err != nil {
		panic(err)
	}
	if ierr != 0 {
		if ierr > 0 && ierr <= n {
			info = ierr
		} else if ierr > n && ierr <= 2*n {
			info = ierr - n
		} else {
			info = n + 1
		}
		goto label130
	}

	//     Compute Eigenvectors and estimate condition numbers if desired
	//     (Workspace: DTGEVC: need 6*N
	//                 DTGSNA: need 2*N*(N+2)+16 if SENSE = 'V' or 'B',
	//                         need N otherwise )
	if ilv || !wantsn {
		if ilv {
			if ilvl {
				if ilvr {
					chtemp = 'B'
				} else {
					chtemp = 'L'
				}
			} else {
				chtemp = 'R'
			}

			if _, ierr, err = Dtgevc(mat.SideByte(chtemp), 'B', ldumma, n, a, b, vl, vr, n, work); err != nil {
				panic(err)
			}
			if ierr != 0 {
				info = n + 2
				goto label130
			}
		}

		if !wantsn {
			//           compute eigenvectors (DTGEVC) and estimate condition
			//           numbers (DTGSNA). Note that the definition of the condition
			//           number is not invariant under transformation (u,v) to
			//           (Q*u, Z*v), where (u,v) are eigenvectors of the generalized
			//           Schur form (S,T), Q and Z are orthogonal matrices. In order
			//           to avoid using extra 2*N*N workspace, we have to recalculate
			//           eigenvectors and estimate one condition numbers at a time.
			pair = false
			for i = 1; i <= n; i++ {

				if pair {
					pair = false
					goto label20
				}
				mm = 1
				if i < n {
					if a.Get(i, i-1) != zero {
						pair = true
						mm = 2
					}
				}

				for j = 1; j <= n; j++ {
					(*bwork)[j-1] = false
				}
				if mm == 1 {
					(*bwork)[i-1] = true
				} else if mm == 2 {
					(*bwork)[i-1] = true
					(*bwork)[i] = true
				}

				iwrk = mm*n + 1
				iwrk1 = iwrk + mm*n

				//              Compute a pair of left and right eigenvectors.
				//              (compute workspace: need up to 4*N + 6*N)
				if wantse || wantsb {
					if _, ierr, err = Dtgevc(Both, 'S', *bwork, n, a, b, work.Matrix(n, opts), work.Off(iwrk-1).Matrix(n, opts), mm, work.Off(iwrk1-1)); err != nil {
						panic(err)
					}
					if ierr != 0 {
						info = n + 2
						goto label130
					}
				}

				if _, err = Dtgsna(sense, 'S', *bwork, n, a, b, work.Matrix(n, opts), work.Off(iwrk-1).Matrix(n, opts), rconde.Off(i-1), rcondv.Off(i-1), mm, work.Off(iwrk1-1), lwork-iwrk1+1, iwork); err != nil {
					panic(err)
				}

			label20:
			}
		}
	}

	//     Undo balancing on VL and VR and normalization
	//     (Workspace: none needed)
	if ilvl {
		if err = Dggbak(balanc, Left, n, ilo, ihi, lscale, rscale, n, vl); err != nil {
			panic(err)
		}

		for jc = 1; jc <= n; jc++ {
			if alphai.Get(jc-1) < zero {
				goto label70
			}
			temp = zero
			if alphai.Get(jc-1) == zero {
				for jr = 1; jr <= n; jr++ {
					temp = math.Max(temp, math.Abs(vl.Get(jr-1, jc-1)))
				}
			} else {
				for jr = 1; jr <= n; jr++ {
					temp = math.Max(temp, math.Abs(vl.Get(jr-1, jc-1))+math.Abs(vl.Get(jr-1, jc)))
				}
			}
			if temp < smlnum {
				goto label70
			}
			temp = one / temp
			if alphai.Get(jc-1) == zero {
				for jr = 1; jr <= n; jr++ {
					vl.Set(jr-1, jc-1, vl.Get(jr-1, jc-1)*temp)
				}
			} else {
				for jr = 1; jr <= n; jr++ {
					vl.Set(jr-1, jc-1, vl.Get(jr-1, jc-1)*temp)
					vl.Set(jr-1, jc, vl.Get(jr-1, jc)*temp)
				}
			}
		label70:
		}
	}
	if ilvr {
		if err = Dggbak(balanc, Right, n, ilo, ihi, lscale, rscale, n, vr); err != nil {
			panic(err)
		}
		for jc = 1; jc <= n; jc++ {
			if alphai.Get(jc-1) < zero {
				goto label120
			}
			temp = zero
			if alphai.Get(jc-1) == zero {
				for jr = 1; jr <= n; jr++ {
					temp = math.Max(temp, math.Abs(vr.Get(jr-1, jc-1)))
				}
			} else {
				for jr = 1; jr <= n; jr++ {
					temp = math.Max(temp, math.Abs(vr.Get(jr-1, jc-1))+math.Abs(vr.Get(jr-1, jc)))
				}
			}
			if temp < smlnum {
				goto label120
			}
			temp = one / temp
			if alphai.Get(jc-1) == zero {
				for jr = 1; jr <= n; jr++ {
					vr.Set(jr-1, jc-1, vr.Get(jr-1, jc-1)*temp)
				}
			} else {
				for jr = 1; jr <= n; jr++ {
					vr.Set(jr-1, jc-1, vr.Get(jr-1, jc-1)*temp)
					vr.Set(jr-1, jc, vr.Get(jr-1, jc)*temp)
				}
			}
		label120:
		}
	}

	//     Undo scaling if necessary
label130:
	;

	if ilascl {
		if err = Dlascl('G', 0, 0, anrmto, anrm, n, 1, alphar.Matrix(n, opts)); err != nil {
			panic(err)
		}
		if err = Dlascl('G', 0, 0, anrmto, anrm, n, 1, alphai.Matrix(n, opts)); err != nil {
			panic(err)
		}
	}

	if ilbscl {
		if err = Dlascl('G', 0, 0, bnrmto, bnrm, n, 1, beta.Matrix(n, opts)); err != nil {
			panic(err)
		}
	}

	work.Set(0, float64(maxwrk))

	return
}
