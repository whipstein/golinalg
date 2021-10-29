package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dggev3 computes for a pair of N-by-N real nonsymmetric matrices (A,B)
// the generalized eigenvalues, and optionally, the left and/or right
// generalized eigenvectors.
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
//                  A * v(j) = lambda(j) * B * v(j).
//
// The left eigenvector u(j) corresponding to the eigenvalue lambda(j)
// of (A,B) satisfies
//
//                  u(j)**H * A  = lambda(j) * u(j)**H * B .
//
// where u(j)**H is the conjugate-transpose of u(j).
func Dggev3(jobvl, jobvr byte, n int, a, b *mat.Matrix, alphar, alphai, beta *mat.Vector, vl, vr *mat.Matrix, work *mat.Vector, lwork int) (info int, err error) {
	var ilascl, ilbscl, ilv, ilvl, ilvr, lquery bool
	var chtemp byte
	var anrm, anrmto, bignum, bnrm, bnrmto, eps, one, smlnum, temp, zero float64
	var icols, ierr, ihi, ijobvl, ijobvr, ileft, ilo, iright, irows, itau, iwrk, jc, jr, lwkopt int

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
		err = fmt.Errorf("vl.Rows < 1 || (ilvl && vl.Rows < n): vl.Rows=%v, ilvl=%v, n=%v", vl.Rows, ilvl, n)
	} else if vr.Rows < 1 || (ilvr && vr.Rows < n) {
		err = fmt.Errorf("vr.Rows < 1 || (ilvr && vr.Rows < n): vr.Rows=%v, ilvr=%v, n=%v", vr.Rows, ilvr, n)
	} else if lwork < max(1, 8*n) && !lquery {
		err = fmt.Errorf("lwork < max(1, 8*n) && !lquery: lwork=%v, n=%v, lquery=%v", lwork, n, lquery)
	}

	//     Compute workspace
	if err == nil {
		if err = Dgeqrf(n, n, b, work, work, -1); err != nil {
			panic(err)
		}
		lwkopt = max(1, 8*n, 3*n+int(work.Get(0)))
		if err = Dormqr(Left, Trans, n, n, n, b, work, a, work, -1); err != nil {
			panic(err)
		}
		lwkopt = max(lwkopt, 3*n+int(work.Get(0)))
		if ilvl {
			if err = Dorgqr(n, n, n, vl, work, work, -1); err != nil {
				panic(err)
			}
			lwkopt = max(lwkopt, 3*n+int(work.Get(0)))
		}
		if ilv {
			if err = Dgghd3(jobvl, jobvr, n, 1, n, a, b, vl, vr, work, -1); err != nil {
				panic(err)
			}
			lwkopt = max(lwkopt, 3*n+int(work.Get(0)))
			if ierr, err = Dhgeqz('S', jobvl, jobvr, n, 1, n, a, b, alphar, alphai, beta, vl, vr, work, -1); err != nil {
				panic(err)
			}
			lwkopt = max(lwkopt, 2*n+int(work.Get(0)))
		} else {
			if err = Dgghd3('N', 'N', n, 1, n, a, b, vl, vr, work, -1); err != nil {
				panic(err)
			}
			lwkopt = max(lwkopt, 3*n+int(work.Get(0)))
			if ierr, err = Dhgeqz('E', jobvl, jobvr, n, 1, n, a, b, alphar, alphai, beta, vl, vr, work, -1); err != nil {
				panic(err)
			}
			lwkopt = max(lwkopt, 2*n+int(work.Get(0)))
		}
		work.Set(0, float64(lwkopt))
	}

	if err != nil {
		gltest.Xerbla2("Dggev3", err)
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

	//     Scale B if max element outside range [SMLNUM,BIGNUM]
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

	//     Permute the matrices A, B to isolate eigenvalues if possible
	ileft = 1
	iright = n + 1
	iwrk = iright + n
	if ilo, ihi, err = Dggbal('P', n, a, b, work.Off(ileft-1), work.Off(iright-1), work.Off(iwrk-1)); err != nil {
		panic(err)
	}

	//     Reduce B to triangular form (QR decomposition of B)
	irows = ihi + 1 - ilo
	if ilv {
		icols = n + 1 - ilo
	} else {
		icols = irows
	}
	itau = iwrk
	iwrk = itau + irows
	if err = Dgeqrf(irows, icols, b.Off(ilo-1, ilo-1), work.Off(itau-1), work.Off(iwrk-1), lwork+1-iwrk); err != nil {
		panic(err)
	}

	//     Apply the orthogonal transformation to matrix A
	if err = Dormqr(Left, Trans, irows, icols, irows, b.Off(ilo-1, ilo-1), work.Off(itau-1), a.Off(ilo-1, ilo-1), work.Off(iwrk-1), lwork+1-iwrk); err != nil {
		panic(err)
	}

	//     Initialize VL
	if ilvl {
		Dlaset(Full, n, n, zero, one, vl)
		if irows > 1 {
			Dlacpy(Lower, irows-1, irows-1, b.Off(ilo, ilo-1), vl.Off(ilo, ilo-1))
		}
		if err = Dorgqr(irows, irows, irows, vl.Off(ilo-1, ilo-1), work.Off(itau-1), work.Off(iwrk-1), lwork+1-iwrk); err != nil {
			panic(err)
		}
	}

	//     Initialize VR
	if ilvr {
		Dlaset(Full, n, n, zero, one, vr)
	}

	//     Reduce to generalized Hessenberg form
	if ilv {
		//        Eigenvectors requested -- work on whole matrix.
		if err = Dgghd3(jobvl, jobvr, n, ilo, ihi, a, b, vl, vr, work.Off(iwrk-1), lwork+1-iwrk); err != nil {
			panic(err)
		}
	} else {
		if err = Dgghd3('N', 'N', irows, 1, irows, a.Off(ilo-1, ilo-1), b.Off(ilo-1, ilo-1), vl, vr, work.Off(iwrk-1), lwork+1-iwrk); err != nil {
			panic(err)
		}
	}

	//     Perform QZ algorithm (Compute eigenvalues, and optionally, the
	//     Schur forms and Schur vectors)
	iwrk = itau
	if ilv {
		chtemp = 'S'
	} else {
		chtemp = 'E'
	}
	if ierr, err = Dhgeqz(chtemp, jobvl, jobvr, n, ilo, ihi, a, b, alphar, alphai, beta, vl, vr, work.Off(iwrk-1), lwork+1-iwrk); err != nil {
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
		goto label110
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
		if _, ierr, err = Dtgevc(mat.SideByte(chtemp), 'B', ldumma, n, a, b, vl, vr, n, work.Off(iwrk-1)); err != nil {
			panic(err)
		}
		if ierr != 0 {
			info = n + 2
			goto label110
		}

		//        Undo balancing on VL and VR and normalization
		if ilvl {
			if err = Dggbak('P', Left, n, ilo, ihi, work.Off(ileft-1), work.Off(iright-1), n, vl); err != nil {
				panic(err)
			}
			for jc = 1; jc <= n; jc++ {
				if alphai.Get(jc-1) < zero {
					goto label50
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
					goto label50
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
			label50:
			}
		}
		if ilvr {
			if err = Dggbak('P', Right, n, ilo, ihi, work.Off(ileft-1), work.Off(iright-1), n, vr); err != nil {
				panic(err)
			}
			for jc = 1; jc <= n; jc++ {
				if alphai.Get(jc-1) < zero {
					goto label100
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
					goto label100
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
			label100:
			}
		}

		//        End of eigenvector calculation
	}

	//     Undo scaling if necessary
label110:
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

	work.Set(0, float64(lwkopt))

	return
}
