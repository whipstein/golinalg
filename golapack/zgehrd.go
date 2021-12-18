package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgehrd reduces a complex general matrix A to upper Hessenberg form H by
// an unitary similarity transformation:  Q**H * A * Q = H .
func Zgehrd(n, ilo, ihi int, a *mat.CMatrix, tau, work *mat.CVector, lwork int) (err error) {
	var lquery bool
	var ei, one, zero complex128
	var i, ib, iwt, j, ldt, ldwork, lwkopt, nb, nbmax, nbmin, nh, nx, tsize int

	nbmax = 64
	ldt = nbmax + 1
	tsize = ldt * nbmax
	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	//     Test the input parameters
	lquery = (lwork == -1)
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if ilo < 1 || ilo > max(1, n) {
		err = fmt.Errorf("ilo < 1 || ilo > max(1, n): ilo=%v, n=%v", ilo, n)
	} else if ihi < min(ilo, n) || ihi > n {
		err = fmt.Errorf("ihi < min(ilo, n) || ihi > n: ilo=%v, ihi=%v, n=%v", ilo, ihi, n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if lwork < max(1, n) && !lquery {
		err = fmt.Errorf("lwork < max(1, n) && !lquery: lwork=%v, n=%v, lquery=%v", lwork, n, lquery)
	}

	if err == nil {
		//        Compute the workspace requirements
		nb = min(nbmax, Ilaenv(1, "Zgehrd", []byte{' '}, n, ilo, ihi, -1))
		lwkopt = n*nb + tsize
		work.SetRe(0, float64(lwkopt))
	}

	if err != nil {
		gltest.Xerbla2("Zgehrd", err)
		return
	} else if lquery {
		return
	}

	//     Set elements 1:ILO-1 and IHI:N-1 of TAU to zero
	for i = 1; i <= ilo-1; i++ {
		tau.Set(i-1, zero)
	}
	for i = max(1, ihi); i <= n-1; i++ {
		tau.Set(i-1, zero)
	}

	//     Quick return if possible
	nh = ihi - ilo + 1
	if nh <= 1 {
		work.Set(0, 1)
		return
	}

	//     Determine the block size
	nb = min(nbmax, Ilaenv(1, "Zgehrd", []byte{' '}, n, ilo, ihi, -1))
	nbmin = 2
	if nb > 1 && nb < nh {
		//        Determine when to cross over from blocked to unblocked code
		//        (last block is always handled by unblocked code)
		nx = max(nb, Ilaenv(3, "Zgehrd", []byte{' '}, n, ilo, ihi, -1))
		if nx < nh {
			//           Determine if workspace is large enough for blocked code
			if lwork < n*nb+tsize {
				//              Not enough workspace to use optimal NB:  determine the
				//              minimum value of NB, and reduce NB or force use of
				//              unblocked code
				nbmin = max(2, Ilaenv(2, "Zgehrd", []byte{' '}, n, ilo, ihi, -1))
				if lwork >= (n*nbmin + tsize) {
					nb = (lwork - tsize) / n
				} else {
					nb = 1
				}
			}
		}
	}
	ldwork = n
	//
	if nb < nbmin || nb >= nh {
		//        Use unblocked code below
		i = ilo

	} else {
		//        Use blocked code
		iwt = 1 + n*nb
		for i = ilo; i <= ihi-1-nx; i += nb {
			ib = min(nb, ihi-i)

			//           Reduce columns i:i+ib-1 to Hessenberg form, returning the
			//           matrices V and T of the block reflector H = I - V*T*V**H
			//           which performs the reduction, and also the matrix Y = A*V*T
			Zlahr2(ihi, i, ib, a.Off(0, i-1), tau.Off(i-1), work.Off(iwt-1).CMatrix(ldt, opts), work.CMatrix(ldwork, opts))

			//           Apply the block reflector H to A(1:ihi,i+ib:ihi) from the
			//           right, computing  A := A - Y * V**H. V(i+ib,ib-1) must be set
			//           to 1
			ei = a.Get(i+ib-1, i+ib-1-1)
			a.Set(i+ib-1, i+ib-1-1, one)
			err = a.Off(0, i+ib-1).Gemm(NoTrans, ConjTrans, ihi, ihi-i-ib+1, ib, -one, work.CMatrix(ldwork, opts), a.Off(i+ib-1, i-1), one)
			a.Set(i+ib-1, i+ib-1-1, ei)

			//           Apply the block reflector H to A(1:i,i+1:i+ib-1) from the
			//           right
			if err = work.CMatrix(ldwork, opts).Trmm(Right, Lower, ConjTrans, Unit, i, ib-1, one, a.Off(i, i-1)); err != nil {
				panic(err)
			}
			for j = 0; j <= ib-2; j++ {
				a.Off(0, i+j).CVector().Axpy(i, -one, work.Off(ldwork*j), 1, 1)
			}

			//           Apply the block reflector H to A(i+1:ihi,i+ib:n) from the
			//           left
			Zlarfb(Left, ConjTrans, 'F', 'C', ihi-i, n-i-ib+1, ib, a.Off(i, i-1), work.Off(iwt-1).CMatrix(ldt, opts), a.Off(i, i+ib-1), work.CMatrix(ldwork, opts))
		}
	}

	//     Use unblocked code to reduce the rest of the matrix
	if err = Zgehd2(n, i, ihi, a, tau, work); err != nil {
		panic(err)
	}
	work.SetRe(0, float64(lwkopt))

	return
}
