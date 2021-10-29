package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dorgtsqr generates an M-by-N real matrix Q_out with orthonormal columns,
// which are the first N columns of a product of real orthogonal
// matrices of order M which are returned by DLATSQR
//
//      Q_out = first_N_columns_of( Q(1)_in * Q(2)_in * ... * Q(k)_in ).
//
// See the documentation for DLATSQR.
func Dorgtsqr(m, n, mb, nb int, a, t *mat.Matrix, work *mat.Vector, lwork int) (err error) {
	var lquery bool
	var one, zero float64
	var j, lc, ldc, lw, lworkopt, nblocal int

	one = 1.0
	zero = 0.0

	//     Test the input parameters
	lquery = lwork == -1
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 || m < n {
		err = fmt.Errorf("n < 0 || m < n: n=%v, m=%v", n, m)
	} else if mb <= n {
		err = fmt.Errorf("mb <= n: mb=%v, n=%v", mb, n)
	} else if nb < 1 {
		err = fmt.Errorf("nb < 1: nb=%v", nb)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if t.Rows < max(1, min(nb, n)) {
		err = fmt.Errorf("t.Rows < max(1, min(nb, n)): t.Rows=%v, nb=%v, n=%v", t.Rows, nb, n)
	} else {
		//        Test the input LWORK for the dimension of the array WORK.
		//        This workspace is used to store array C(LDC, N) and WORK(LWORK)
		//        in the call to DLAMTSQR. See the documentation for DLAMTSQR.
		if lwork < 2 && !lquery {
			err = fmt.Errorf("lwork < 2 && !lquery: lwork=%v, lquery=%v", lwork, lquery)
		} else {
			//           Set block size for column blocks
			nblocal = min(nb, n)

			//           LWORK = -1, then set the size for the array C(LDC,N)
			//           in DLAMTSQR call and set the optimal size of the work array
			//           WORK(LWORK) in DLAMTSQR call.
			ldc = m
			lc = ldc * n
			lw = n * nblocal

			lworkopt = lc + lw

			if lwork < max(1, lworkopt) && !lquery {
				err = fmt.Errorf("lwork < max(1, lworkopt) && !lquery: lwork=%v, lworkopt=%v, lquery=%v", lwork, lworkopt, lquery)
			}
		}

	}

	//     Handle error in the input parameters and return workspace query.
	if err != nil {
		gltest.Xerbla2("Dorgtsqr", err)
		return
	} else if lquery {
		work.Set(0, float64(lworkopt))
		return
	}

	//     Quick return if possible
	if min(m, n) == 0 {
		work.Set(0, float64(lworkopt))
		return
	}

	//     (1) Form explicitly the tall-skinny M-by-N left submatrix Q1_in
	//     of M-by-M orthogonal matrix Q_in, which is implicitly stored in
	//     the subdiagonal part of input array A and in the input array T.
	//     Perform by the following operation using the routine DLAMTSQR.
	//
	//         Q1_in = Q_in * ( I ), where I is a N-by-N identity matrix,
	//                        ( 0 )        0 is a (M-N)-by-N zero matrix.
	//
	//     (1a) Form M-by-N matrix in the array WORK(1:LDC*N) with ones
	//     on the diagonal and zeros elsewhere.
	Dlaset(Full, m, n, zero, one, work.Matrix(ldc, opts))

	//     (1b)  On input, WORK(1:LDC*N) stores ( I );
	//                                          ( 0 )
	//
	//           On output, WORK(1:LDC*N) stores Q1_in.
	if err = Dlamtsqr(Left, NoTrans, m, n, n, mb, nblocal, a, t, work.Matrix(ldc, opts), work.Off(lc), lw); err != nil {
		panic(err)
	}

	//     (2) Copy the result from the part of the work array (1:M,1:N)
	//     with the leading dimension LDC that starts at WORK(1) into
	//     the output array A(1:M,1:N) column-by-column.
	for j = 1; j <= n; j++ {
		goblas.Dcopy(m, work.Off((j-1)*ldc, 1), a.Vector(0, j-1, 1))
	}

	work.Set(0, float64(lworkopt))

	return
}
