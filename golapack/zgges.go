package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgges computes for a pair of N-by-N complex nonsymmetric matrices
// (A,B), the generalized eigenvalues, the generalized complex Schur
// form (S, T), and optionally left and/or right Schur vectors (VSL
// and VSR). This gives the generalized Schur factorization
//
//         (A,B) = ( (VSL)*S*(VSR)**H, (VSL)*T*(VSR)**H )
//
// where (VSR)**H is the conjugate-transpose of VSR.
//
// Optionally, it also orders the eigenvalues so that a selected cluster
// of eigenvalues appears in the leading diagonal blocks of the upper
// triangular matrix S and the upper triangular matrix T. The leading
// columns of VSL and VSR then form an unitary basis for the
// corresponding left and right eigenspaces (deflating subspaces).
//
// (If only the generalized eigenvalues are needed, use the driver
// ZGGEV instead, which is faster.)
//
// A generalized eigenvalue for a pair of matrices (A,B) is a scalar w
// or a ratio alpha/beta = w, such that  A - w*B is singular.  It is
// usually represented as the pair (alpha,beta), as there is a
// reasonable interpretation for beta=0, and even for both being zero.
//
// A pair of matrices (S,T) is in generalized complex Schur form if S
// and T are upper triangular and, in addition, the diagonal elements
// of T are non-negative real numbers.
func Zgges(jobvsl, jobvsr, sort byte, selctg func(complex128, complex128) bool, n int, a, b *mat.CMatrix, alpha, beta *mat.CVector, vsl, vsr *mat.CMatrix, work *mat.CVector, lwork int, rwork *mat.Vector, bwork *[]bool) (sdim, info int, err error) {
	var cursl, ilascl, ilbscl, ilvsl, ilvsr, lastsl, lquery, wantst bool
	var cone, czero complex128
	var anrm, anrmto, bignum, bnrm, bnrmto, eps, one, smlnum, zero float64
	var i, icols, ierr, ihi, ijobvl, ijobvr, ileft, ilo, iright, irows, irwrk, itau, iwrk, lwkmin, lwkopt int

	dif := vf(2)
	idum := make([]int, 1)

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Decode the input arguments
	if jobvsl == 'N' {
		ijobvl = 1
		ilvsl = false
	} else if jobvsl == 'V' {
		ijobvl = 2
		ilvsl = true
	} else {
		ijobvl = -1
		ilvsl = false
	}

	if jobvsr == 'N' {
		ijobvr = 1
		ilvsr = false
	} else if jobvsr == 'V' {
		ijobvr = 2
		ilvsr = true
	} else {
		ijobvr = -1
		ilvsr = false
	}

	wantst = sort == 'S'

	//     Test the input arguments
	lquery = (lwork == -1)
	if ijobvl <= 0 {
		err = fmt.Errorf("ijobvl <= 0: jobvsl='%c'", jobvsl)
	} else if ijobvr <= 0 {
		err = fmt.Errorf("ijobvr <= 0: jobvsr='%c'", jobvsr)
	} else if (!wantst) && (sort != 'N') {
		err = fmt.Errorf("(!wantst) && (sort != 'N'): sort='%c'", sort)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	} else if vsl.Rows < 1 || (ilvsl && vsl.Rows < n) {
		err = fmt.Errorf("vsl.Rows < 1 || (ilvsl && vsl.Rows < n): vsl.Rows=%v, n=%v, ilvsl=%v", vsl.Rows, n, ilvsl)
	} else if vsr.Rows < 1 || (ilvsr && vsr.Rows < n) {
		err = fmt.Errorf("vsr.Rows < 1 || (ilvsr && vsr.Rows < n): vsr.Rows=%v, n=%v, ilvsr=%v", vsr.Rows, n, ilvsr)
	}

	//     Compute workspace
	//      (Note: Comments in the code beginning "Workspace:" describe the
	//       minimal amount of workspace needed at that point in the code,
	//       as well as the preferred amount for good performance.
	//       NB refers to the optimal block size for the immediately
	//       following subroutine, as returned by ILAENV.)
	if err == nil {
		lwkmin = max(1, 2*n)
		lwkopt = max(1, n+n*Ilaenv(1, "Zgeqrf", []byte{' '}, n, 1, n, 0))
		lwkopt = max(lwkopt, n+n*Ilaenv(1, "Zunmqr", []byte{' '}, n, 1, n, -1))
		if ilvsl {
			lwkopt = max(lwkopt, n+n*Ilaenv(1, "Zungqr", []byte{' '}, n, 1, n, -1))
		}
		work.SetRe(0, float64(lwkopt))

		if lwork < lwkmin && !lquery {
			err = fmt.Errorf("lwork < lwkmin && !lquery: lwork=%v, lwkmin=%v, lquery=%v", lwork, lwkmin, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Zgges", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
		sdim = 0
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
	//
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

	//     Permute the matrix to make it more nearly triangular
	//     (Real Workspace: need 6*N)
	ileft = 1
	iright = n + 1
	irwrk = iright + n
	if ilo, ihi, err = Zggbal('P', n, a, b, rwork.Off(ileft-1), rwork.Off(iright-1), rwork.Off(irwrk-1)); err != nil {
		panic(err)
	}

	//     Reduce B to triangular form (QR decomposition of B)
	//     (Complex Workspace: need N, prefer N*NB)
	irows = ihi + 1 - ilo
	icols = n + 1 - ilo
	itau = 1
	iwrk = itau + irows
	if err = Zgeqrf(irows, icols, b.Off(ilo-1, ilo-1), work.Off(itau-1), work.Off(iwrk-1), lwork+1-iwrk); err != nil {
		panic(err)
	}

	//     Apply the orthogonal transformation to matrix A
	//     (Complex Workspace: need N, prefer N*NB)
	if err = Zunmqr(Left, ConjTrans, irows, icols, irows, b.Off(ilo-1, ilo-1), work.Off(itau-1), a.Off(ilo-1, ilo-1), work.Off(iwrk-1), lwork+1-iwrk); err != nil {
		panic(err)
	}

	//     Initialize VSL
	//     (Complex Workspace: need N, prefer N*NB)
	if ilvsl {
		Zlaset(Full, n, n, czero, cone, vsl)
		if irows > 1 {
			Zlacpy(Lower, irows-1, irows-1, b.Off(ilo, ilo-1), vsl.Off(ilo, ilo-1))
		}
		if err = Zungqr(irows, irows, irows, vsl.Off(ilo-1, ilo-1), work.Off(itau-1), work.Off(iwrk-1), lwork+1-iwrk); err != nil {
			panic(err)
		}
	}

	//     Initialize VSR
	if ilvsr {
		Zlaset(Full, n, n, czero, cone, vsr)
	}

	//     Reduce to generalized Hessenberg form
	//     (Workspace: none needed)
	if err = Zgghrd(jobvsl, jobvsr, n, ilo, ihi, a, b, vsl, vsr); err != nil {
		panic(err)
	}

	sdim = 0

	//     Perform QZ algorithm, computing Schur vectors if desired
	//     (Complex Workspace: need N)
	//     (Real Workspace: need N)
	iwrk = itau
	if ierr, err = Zhgeqz('S', jobvsl, jobvsr, n, ilo, ihi, a, b, alpha, beta, vsl, vsr, work.Off(iwrk-1), lwork+1-iwrk, rwork.Off(irwrk-1)); err != nil {
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
		goto label30
	}

	//     Sort eigenvalues ALPHA/BETA if desired
	//     (Workspace: none needed)
	if wantst {
		//        Undo scaling on eigenvalues before selecting
		if ilascl {
			if err = Zlascl('G', 0, 0, anrm, anrmto, n, 1, alpha.CMatrix(n, opts)); err != nil {
				panic(err)
			}
		}
		if ilbscl {
			if err = Zlascl('G', 0, 0, bnrm, bnrmto, n, 1, beta.CMatrix(n, opts)); err != nil {
				panic(err)
			}
		}

		//        Select eigenvalues
		for i = 1; i <= n; i++ {
			(*bwork)[i-1] = selctg(alpha.Get(i-1), beta.Get(i-1))
		}
		//
		if sdim, _, _, ierr, err = Ztgsen(0, ilvsl, ilvsr, *bwork, n, a, b, alpha, beta, vsl, vsr, dif, work.Off(iwrk-1), lwork-iwrk+1, &idum, 1); err != nil || ierr == 1 {
			info = n + 3
		}

	}

	//     Apply back-permutation to VSL and VSR
	//     (Workspace: none needed)
	if ilvsl {
		if err = Zggbak('P', Left, n, ilo, ihi, rwork.Off(ileft-1), rwork.Off(iright-1), n, vsl); err != nil {
			panic(err)
		}
	}
	if ilvsr {
		if err = Zggbak('P', Right, n, ilo, ihi, rwork.Off(ileft-1), rwork.Off(iright-1), n, vsr); err != nil {
			panic(err)
		}
	}

	//     Undo scaling
	if ilascl {
		if err = Zlascl('U', 0, 0, anrmto, anrm, n, n, a); err != nil {
			panic(err)
		}
		if err = Zlascl('G', 0, 0, anrmto, anrm, n, 1, alpha.CMatrix(n, opts)); err != nil {
			panic(err)
		}
	}

	if ilbscl {
		if err = Zlascl('U', 0, 0, bnrmto, bnrm, n, n, b); err != nil {
			panic(err)
		}
		if err = Zlascl('G', 0, 0, bnrmto, bnrm, n, 1, beta.CMatrix(n, opts)); err != nil {
			panic(err)
		}
	}

	if wantst {
		//        Check if reordering is correct
		lastsl = true
		sdim = 0
		for i = 1; i <= n; i++ {
			cursl = selctg(alpha.Get(i-1), beta.Get(i-1))
			if cursl {
				sdim = sdim + 1
			}
			if cursl && !lastsl {
				info = n + 2
			}
			lastsl = cursl
		}

	}

label30:
	;

	work.SetRe(0, float64(lwkopt))

	return
}
