package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zggevx computes for a pair of N-by-N complex nonsymmetric matrices
// (A,B) the generalized eigenvalues, and optionally, the left and/or
// right generalized eigenvectors.
//
// Optionally, it also computes a balancing transformation to improve
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
//                  A * v(j) = lambda(j) * B * v(j) .
// The left eigenvector u(j) corresponding to the eigenvalue lambda(j)
// of (A,B) satisfies
//                  u(j)**H * A  = lambda(j) * u(j)**H * B.
// where u(j)**H is the conjugate-transpose of u(j).
func Zggevx(balanc, jobvl, jobvr, sense byte, n int, a, b *mat.CMatrix, alpha, beta *mat.CVector, vl, vr *mat.CMatrix, lscale, rscale, rconde, rcondv *mat.Vector, work *mat.CVector, lwork int, rwork *mat.Vector, iwork *[]int, bwork *[]bool) (ilo, ihi int, abnrm, bbnrm float64, info int, err error) {
	var ilascl, ilbscl, ilv, ilvl, ilvr, lquery, noscl, wantsb, wantse, wantsn, wantsv bool
	var chtemp byte
	var cone, czero complex128
	var anrm, anrmto, bignum, bnrm, bnrmto, eps, one, smlnum, temp, zero float64
	var i, icols, ierr, ijobvl, ijobvr, irows, itau, iwrk, iwrk1, j, jc, jr, maxwrk, minwrk int

	ldumma := make([]bool, 1)

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

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
	if !(noscl || balanc == 'S' || balanc == 'B') {
		err = fmt.Errorf("!(noscl || balanc == 'S' || balanc == 'B'): balanc='%c'", balanc)
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
		err = fmt.Errorf("vl.Rows < 1 || (ilvl && vl.Rows < n): vl.Rows=%v, n=%v, ilvl=%v", vl.Rows, n, ilvl)
	} else if vr.Rows < 1 || (ilvr && vr.Rows < n) {
		err = fmt.Errorf("vr.Rows < 1 || (ilvr && vr.Rows < n): vr.Rows=%v, n=%v, ilvr=%v", vr.Rows, n, ilvr)
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
			minwrk = 2 * n
			if wantse {
				minwrk = 4 * n
			} else if wantsv || wantsb {
				minwrk = 2 * n * (n + 1)
			}
			maxwrk = minwrk
			maxwrk = max(maxwrk, n+n*Ilaenv(1, "Zgeqrf", []byte{' '}, n, 1, n, 0))
			maxwrk = max(maxwrk, n+n*Ilaenv(1, "Zunmqr", []byte{' '}, n, 1, n, 0))
			if ilvl {
				maxwrk = max(maxwrk, n+n*Ilaenv(1, "Zungqr", []byte{' '}, n, 1, n, 0))
			}
		}
		work.SetRe(0, float64(maxwrk))

		if lwork < minwrk && !lquery {
			err = fmt.Errorf("lwork < minwrk && !lquery: lwork=%v, minwrk=%v, lquery=%v", lwork, minwrk, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Zggevx", err)
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

	//     Scale A if max element outside range [SMLNUM,BIGNUM]
	anrm = Zlange('M', n, n, a, rwork)
	ilascl = false
	if anrm > zero && anrm < smlnum {
		anrmto = smlnum
		ilascl = true
	} else if anrm > bignum {
		anrmto = bignum
		ilascl = true
	}
	if ilascl {
		if err = Zlascl('G', 0, 0, anrm, anrmto, n, n, a); err != nil {
			panic(err)
		}
	}

	//     Scale B if max element outside range [SMLNUM,BIGNUM]
	bnrm = Zlange('M', n, n, b, rwork)
	ilbscl = false
	if bnrm > zero && bnrm < smlnum {
		bnrmto = smlnum
		ilbscl = true
	} else if bnrm > bignum {
		bnrmto = bignum
		ilbscl = true
	}
	if ilbscl {
		if err = Zlascl('G', 0, 0, bnrm, bnrmto, n, n, b); err != nil {
			panic(err)
		}
	}

	//     Permute and/or balance the matrix pair (A,B)
	//     (Real Workspace: need 6*N if BALANC = 'S' or 'B', 1 otherwise)
	if ilo, ihi, err = Zggbal(balanc, n, a, b, lscale, rscale, rwork); err != nil {
		panic(err)
	}

	//     Compute ABNRM and BBNRM
	abnrm = Zlange('1', n, n, a, rwork.Off(0))
	if ilascl {
		rwork.Set(0, abnrm)
		if err = Dlascl('G', 0, 0, anrmto, anrm, 1, 1, rwork.Matrix(1, opts)); err != nil {
			panic(err)
		}
		abnrm = rwork.Get(0)
	}

	bbnrm = Zlange('1', n, n, b, rwork.Off(0))
	if ilbscl {
		rwork.Set(0, bbnrm)
		if err = Dlascl('G', 0, 0, bnrmto, bnrm, 1, 1, rwork.Matrix(1, opts)); err != nil {
			panic(err)
		}
		bbnrm = rwork.Get(0)
	}

	//     Reduce B to triangular form (QR decomposition of B)
	//     (Complex Workspace: need N, prefer N*NB )
	irows = ihi + 1 - ilo
	if ilv || !wantsn {
		icols = n + 1 - ilo
	} else {
		icols = irows
	}
	itau = 1
	iwrk = itau + irows
	if err = Zgeqrf(irows, icols, b.Off(ilo-1, ilo-1), work.Off(itau-1), work.Off(iwrk-1), lwork+1-iwrk); err != nil {
		panic(err)
	}

	//     Apply the unitary transformation to A
	//     (Complex Workspace: need N, prefer N*NB)
	if err = Zunmqr(Left, ConjTrans, irows, icols, irows, b.Off(ilo-1, ilo-1), work.Off(itau-1), a.Off(ilo-1, ilo-1), work.Off(iwrk-1), lwork+1-iwrk); err != nil {
		panic(err)
	}

	//     Initialize VL and/or VR
	//     (Workspace: need N, prefer N*NB)
	if ilvl {
		Zlaset(Full, n, n, czero, cone, vl)
		if irows > 1 {
			Zlacpy(Lower, irows-1, irows-1, b.Off(ilo, ilo-1), vl.Off(ilo, ilo-1))
		}
		if err = Zungqr(irows, irows, irows, vl.Off(ilo-1, ilo-1), work.Off(itau-1), work.Off(iwrk-1), lwork+1-iwrk); err != nil {
			panic(err)
		}
	}

	if ilvr {
		Zlaset(Full, n, n, czero, cone, vr)
	}

	//     Reduce to generalized Hessenberg form
	//     (Workspace: none needed)
	if ilv || !wantsn {
		//        Eigenvectors requested -- work on whole matrix.
		if err = Zgghrd(jobvl, jobvr, n, ilo, ihi, a, b, vl, vr); err != nil {
			panic(err)
		}
	} else {
		if err = Zgghrd('N', 'N', irows, 1, irows, a.Off(ilo-1, ilo-1), b.Off(ilo-1, ilo-1), vl, vr); err != nil {
			panic(err)
		}
	}

	//     Perform QZ algorithm (Compute eigenvalues, and optionally, the
	//     Schur forms and Schur vectors)
	//     (Complex Workspace: need N)
	//     (Real Workspace: need N)
	iwrk = itau
	if ilv || !wantsn {
		chtemp = 'S'
	} else {
		chtemp = 'E'
	}

	if ierr, err = Zhgeqz(chtemp, jobvl, jobvr, n, ilo, ihi, a, b, alpha, beta, vl, vr, work.Off(iwrk-1), lwork+1-iwrk, rwork); err != nil {
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
		goto label90
	}

	//     Compute Eigenvectors and estimate condition numbers if desired
	//     ZTGEVC: (Complex Workspace: need 2*N )
	//             (Real Workspace:    need 2*N )
	//     ZTGSNA: (Complex Workspace: need 2*N*N if SENSE='V' or 'B')
	//             (Integer Workspace: need N+2 )
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

			if _, err = Ztgevc(mat.SideByte(chtemp), 'B', ldumma, n, a, b, vl, vr, n, work.Off(iwrk-1), rwork); err != nil {
				info = n + 2
				goto label90
			}
		}

		if !wantsn {
			//           compute eigenvectors (DTGEVC) and estimate condition
			//           numbers (DTGSNA). Note that the definition of the condition
			//           number is not invariant under transformation (u,v) to
			//           (Q*u, Z*v), where (u,v) are eigenvectors of the generalized
			//           Schur form (S,T), Q and Z are orthogonal matrices. In order
			//           to avoid using extra 2*N*N workspace, we have to
			//           re-calculate eigenvectors and estimate the condition numbers
			//           one at a time.
			for i = 1; i <= n; i++ {

				for j = 1; j <= n; j++ {
					(*bwork)[j-1] = false
				}
				(*bwork)[i-1] = true

				iwrk = n + 1
				iwrk1 = iwrk + n

				if wantse || wantsb {
					if _, err = Ztgevc(Both, 'S', *bwork, n, a, b, work.CMatrix(n, opts), work.Off(iwrk-1).CMatrix(n, opts), 1, work.Off(iwrk1-1), rwork); err != nil {
						info = n + 2
						goto label90
					}
				}

				if _, err = Ztgsna(sense, 'S', *bwork, n, a, b, work.CMatrix(n, opts), work.Off(iwrk-1).CMatrix(n, opts), rconde.Off(i-1), rcondv.Off(i-1), 1, work.Off(iwrk1-1), lwork-iwrk1+1, iwork); err != nil {
					panic(err)
				}

			}
		}
	}

	//     Undo balancing on VL and VR and normalization
	//     (Workspace: none needed)
	if ilvl {
		if err = Zggbak(balanc, Left, n, ilo, ihi, lscale, rscale, n, vl); err != nil {
			panic(err)
		}

		for jc = 1; jc <= n; jc++ {
			temp = zero
			for jr = 1; jr <= n; jr++ {
				temp = math.Max(temp, abs1(vl.Get(jr-1, jc-1)))
			}
			if temp < smlnum {
				goto label50
			}
			temp = one / temp
			for jr = 1; jr <= n; jr++ {
				vl.Set(jr-1, jc-1, vl.Get(jr-1, jc-1)*toCmplx(temp))
			}
		label50:
		}
	}

	if ilvr {
		if err = Zggbak(balanc, Right, n, ilo, ihi, lscale, rscale, n, vr); err != nil {
			panic(err)
		}
		for jc = 1; jc <= n; jc++ {
			temp = zero
			for jr = 1; jr <= n; jr++ {
				temp = math.Max(temp, abs1(vr.Get(jr-1, jc-1)))
			}
			if temp < smlnum {
				goto label80
			}
			temp = one / temp
			for jr = 1; jr <= n; jr++ {
				vr.Set(jr-1, jc-1, vr.Get(jr-1, jc-1)*toCmplx(temp))
			}
		label80:
		}
	}

	//     Undo scaling if necessary

label90:
	;

	if ilascl {
		if err = Zlascl('G', 0, 0, anrmto, anrm, n, 1, alpha.CMatrix(n, opts)); err != nil {
			panic(err)
		}
	}

	if ilbscl {
		if err = Zlascl('G', 0, 0, bnrmto, bnrm, n, 1, beta.CMatrix(n, opts)); err != nil {
			panic(err)
		}
	}

	work.SetRe(0, float64(maxwrk))

	return
}
