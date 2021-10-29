package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zggev3 computes for a pair of N-by-N complex nonsymmetric matrices
// (A,B), the generalized eigenvalues, and optionally, the left and/or
// right generalized eigenvectors.
//
// A generalized eigenvalue for a pair of matrices (A,B) is a scalar
// lambda or a ratio alpha/beta = lambda, such that A - lambda*B is
// singular. It is usually represented as the pair (alpha,beta), as
// there is a reasonable interpretation for beta=0, and even for both
// being zero.
//
// The right generalized eigenvector v(j) corresponding to the
// generalized eigenvalue lambda(j) of (A,B) satisfies
//
//              A * v(j) = lambda(j) * B * v(j).
//
// The left generalized eigenvector u(j) corresponding to the
// generalized eigenvalues lambda(j) of (A,B) satisfies
//
//              u(j)**H * A = lambda(j) * u(j)**H * B
//
// where u(j)**H is the conjugate-transpose of u(j).
func Zggev3(jobvl, jobvr byte, n int, a, b *mat.CMatrix, alpha, beta *mat.CVector, vl, vr *mat.CMatrix, work *mat.CVector, lwork int, rwork *mat.Vector) (info int, err error) {
	var ilascl, ilbscl, ilv, ilvl, ilvr, lquery bool
	var chtemp byte
	var cone, czero complex128
	var anrm, anrmto, bignum, bnrm, bnrmto, eps, one, smlnum, temp, zero float64
	var icols, ierr, ihi, ijobvl, ijobvr, ileft, ilo, iright, irows, irwrk, itau, iwrk, jc, jr, lwkopt int

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

	//     Test the input arguments
	lquery = (lwork == -1)
	if ijobvl <= 0 {
		err = fmt.Errorf("ijobvl <= 0: jobvl='%c'", jobvl)
	} else if ijobvr <= 0 {
		err = fmt.Errorf("ijobvr <= 0: jobvr='%c'", jobvr)
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
	} else if lwork < max(1, 2*n) && !lquery {
		err = fmt.Errorf("lwork < max(1, 2*n) && !lquery: lwork=%v, n=%v, lquery=%v", lwork, n, lquery)
	}

	//     Compute workspace
	if err == nil {
		if err = Zgeqrf(n, n, b, work, work, -1); err != nil {
			panic(err)
		}
		lwkopt = max(1, n+int(work.GetRe(0)))
		if err = Zunmqr(Left, ConjTrans, n, n, n, b, work, a, work, -1); err != nil {
			panic(err)
		}
		lwkopt = max(lwkopt, n+int(work.GetRe(0)))
		if ilvl {
			if err = Zungqr(n, n, n, vl, work, work, -1); err != nil {
				panic(err)
			}
			lwkopt = max(lwkopt, n+int(work.GetRe(0)))
		}
		if ilv {
			if err = Zgghd3(jobvl, jobvr, n, 1, n, a, b, vl, vr, work, -1); err != nil {
				panic(err)
			}
			lwkopt = max(lwkopt, n+int(work.GetRe(0)))
			if ierr, err = Zhgeqz('S', jobvl, jobvr, n, 1, n, a, b, alpha, beta, vl, vr, work, -1, rwork); err != nil {
				panic(err)
			}
			lwkopt = max(lwkopt, n+int(work.GetRe(0)))
		} else {
			if err = Zgghd3(jobvl, jobvr, n, 1, n, a, b, vl, vr, work, -1); err != nil {
				panic(err)
			}
			lwkopt = max(lwkopt, n+int(work.GetRe(0)))
			if ierr, err = Zhgeqz('E', jobvl, jobvr, n, 1, n, a, b, alpha, beta, vl, vr, work, -1, rwork); err != nil {
				panic(err)
			}
			lwkopt = max(lwkopt, n+int(work.GetRe(0)))
		}
		work.SetRe(0, float64(lwkopt))
	}

	if err != nil {
		gltest.Xerbla2("Zggev3", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Get machine constants
	eps = Dlamch(Epsilon) * Dlamch(Base)
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

	//     Permute the matrices A, B to isolate eigenvalues if possible
	ileft = 1
	iright = n + 1
	irwrk = iright + n
	if ilo, ihi, err = Zggbal('P', n, a, b, rwork.Off(ileft-1), rwork.Off(iright-1), rwork.Off(irwrk-1)); err != nil {
		panic(err)
	}

	//     Reduce B to triangular form (QR decomposition of B)
	irows = ihi + 1 - ilo
	if ilv {
		icols = n + 1 - ilo
	} else {
		icols = irows
	}
	itau = 1
	iwrk = itau + irows
	if err = Zgeqrf(irows, icols, b.Off(ilo-1, ilo-1), work.Off(itau-1), work.Off(iwrk-1), lwork+1-iwrk); err != nil {
		panic(err)
	}

	//     Apply the orthogonal transformation to matrix A
	if err = Zunmqr(Left, ConjTrans, irows, icols, irows, b.Off(ilo-1, ilo-1), work.Off(itau-1), a.Off(ilo-1, ilo-1), work.Off(iwrk-1), lwork+1-iwrk); err != nil {
		panic(err)
	}

	//     Initialize VL
	if ilvl {
		Zlaset(Full, n, n, czero, cone, vl)
		if irows > 1 {
			Zlacpy(Lower, irows-1, irows-1, b.Off(ilo, ilo-1), vl.Off(ilo, ilo-1))
		}
		if err = Zungqr(irows, irows, irows, vl.Off(ilo-1, ilo-1), work.Off(itau-1), work.Off(iwrk-1), lwork+1-iwrk); err != nil {
			panic(err)
		}
	}

	//     Initialize VR
	if ilvr {
		Zlaset(Full, n, n, czero, cone, vr)
	}

	//     Reduce to generalized Hessenberg form
	if ilv {
		//        Eigenvectors requested -- work on whole matrix.
		if err = Zgghd3(jobvl, jobvr, n, ilo, ihi, a, b, vl, vr, work.Off(iwrk-1), lwork+1-iwrk); err != nil {
			panic(err)
		}
	} else {
		if err = Zgghd3('N', 'N', irows, 1, irows, a.Off(ilo-1, ilo-1), b.Off(ilo-1, ilo-1), vl, vr, work.Off(iwrk-1), lwork+1-iwrk); err != nil {
			panic(err)
		}
	}

	//     Perform QZ algorithm (Compute eigenvalues, and optionally, the
	//     Schur form and Schur vectors)
	iwrk = itau
	if ilv {
		chtemp = 'S'
	} else {
		chtemp = 'E'
	}
	if ierr, err = Zhgeqz(chtemp, jobvl, jobvr, n, ilo, ihi, a, b, alpha, beta, vl, vr, work.Off(iwrk-1), lwork+1-iwrk, rwork.Off(irwrk-1)); err != nil {
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
		goto label70
	}

	//     Compute Eigenvectors
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

		if _, err = Ztgevc(mat.SideByte(chtemp), 'B', ldumma, n, a, b, vl, vr, n, work.Off(iwrk-1), rwork.Off(irwrk-1)); err != nil {
			info = n + 2
			goto label70
		}

		//        Undo balancing on VL and VR and normalization
		if ilvl {
			if err = Zggbak('P', Left, n, ilo, ihi, rwork.Off(ileft-1), rwork.Off(iright-1), n, vl); err != nil {
				panic(err)
			}
			for jc = 1; jc <= n; jc++ {
				temp = zero
				for jr = 1; jr <= n; jr++ {
					temp = math.Max(temp, abs1(vl.Get(jr-1, jc-1)))
				}
				if temp < smlnum {
					goto label30
				}
				temp = one / temp
				for jr = 1; jr <= n; jr++ {
					vl.Set(jr-1, jc-1, vl.Get(jr-1, jc-1)*toCmplx(temp))
				}
			label30:
			}
		}
		if ilvr {
			if err = Zggbak('P', Right, n, ilo, ihi, rwork.Off(ileft-1), rwork.Off(iright-1), n, vr); err != nil {
				panic(err)
			}
			for jc = 1; jc <= n; jc++ {
				temp = zero
				for jr = 1; jr <= n; jr++ {
					temp = math.Max(temp, abs1(vr.Get(jr-1, jc-1)))
				}
				if temp < smlnum {
					goto label60
				}
				temp = one / temp
				for jr = 1; jr <= n; jr++ {
					vr.Set(jr-1, jc-1, vr.Get(jr-1, jc-1)*toCmplx(temp))
				}
			label60:
			}
		}
	}

	//     Undo scaling if necessary
label70:
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

	work.SetRe(0, float64(lwkopt))

	return
}
