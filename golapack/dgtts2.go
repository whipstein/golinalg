package golapack

import "github.com/whipstein/golinalg/mat"

// Dgtts2 solves one of the systems of equations
//    A*X = B  or  A**T*X = B,
// with a tridiagonal matrix A using the LU factorization computed
// by DGTTRF.
func Dgtts2(itrans, n, nrhs int, dl, d, du, du2 *mat.Vector, ipiv []int, b *mat.Matrix) {
	var temp float64
	var i, ip, j int

	//     Quick return if possible
	if n == 0 || nrhs == 0 {
		return
	}

	if itrans == 0 {
		//        Solve A*X = B using the LU factorization of A,
		//        overwriting each right hand side vector with its solution.
		if nrhs <= 1 {
			j = 1
		label10:
			;

			//           Solve L*x = b.
			for i = 1; i <= n-1; i++ {
				ip = ipiv[i-1]
				temp = b.Get(i+1-ip+i-1, j-1) - dl.Get(i-1)*b.Get(ip-1, j-1)
				b.Set(i-1, j-1, b.Get(ip-1, j-1))
				b.Set(i, j-1, temp)
			}

			//           Solve U*x = b.
			b.Set(n-1, j-1, b.Get(n-1, j-1)/d.Get(n-1))
			if n > 1 {
				b.Set(n-1-1, j-1, (b.Get(n-1-1, j-1)-du.Get(n-1-1)*b.Get(n-1, j-1))/d.Get(n-1-1))
			}
			for i = n - 2; i >= 1; i-- {
				b.Set(i-1, j-1, (b.Get(i-1, j-1)-du.Get(i-1)*b.Get(i, j-1)-du2.Get(i-1)*b.Get(i+2-1, j-1))/d.Get(i-1))
			}
			if j < nrhs {
				j = j + 1
				goto label10
			}
		} else {
			for j = 1; j <= nrhs; j++ {
				//              Solve L*x = b.
				for i = 1; i <= n-1; i++ {
					if ipiv[i-1] == i {
						b.Set(i, j-1, b.Get(i, j-1)-dl.Get(i-1)*b.Get(i-1, j-1))
					} else {
						temp = b.Get(i-1, j-1)
						b.Set(i-1, j-1, b.Get(i, j-1))
						b.Set(i, j-1, temp-dl.Get(i-1)*b.Get(i-1, j-1))
					}
				}

				//              Solve U*x = b.
				b.Set(n-1, j-1, b.Get(n-1, j-1)/d.Get(n-1))
				if n > 1 {
					b.Set(n-1-1, j-1, (b.Get(n-1-1, j-1)-du.Get(n-1-1)*b.Get(n-1, j-1))/d.Get(n-1-1))
				}
				for i = n - 2; i >= 1; i-- {
					b.Set(i-1, j-1, (b.Get(i-1, j-1)-du.Get(i-1)*b.Get(i, j-1)-du2.Get(i-1)*b.Get(i+2-1, j-1))/d.Get(i-1))
				}
			}
		}
	} else {
		//        Solve A**T * X = B.
		if nrhs <= 1 {
			//           Solve U**T*x = b.
			j = 1
		label70:
			;
			b.Set(0, j-1, b.Get(0, j-1)/d.Get(0))
			if n > 1 {
				b.Set(1, j-1, (b.Get(1, j-1)-du.Get(0)*b.Get(0, j-1))/d.Get(1))
			}
			for i = 3; i <= n; i++ {
				b.Set(i-1, j-1, (b.Get(i-1, j-1)-du.Get(i-1-1)*b.Get(i-1-1, j-1)-du2.Get(i-2-1)*b.Get(i-2-1, j-1))/d.Get(i-1))
			}

			//           Solve L**T*x = b.
			for i = n - 1; i >= 1; i-- {
				ip = ipiv[i-1]
				temp = b.Get(i-1, j-1) - dl.Get(i-1)*b.Get(i, j-1)
				b.Set(i-1, j-1, b.Get(ip-1, j-1))
				b.Set(ip-1, j-1, temp)
			}
			if j < nrhs {
				j = j + 1
				goto label70
			}

		} else {
			for j = 1; j <= nrhs; j++ {
				//              Solve U**T*x = b.
				b.Set(0, j-1, b.Get(0, j-1)/d.Get(0))
				if n > 1 {
					b.Set(1, j-1, (b.Get(1, j-1)-du.Get(0)*b.Get(0, j-1))/d.Get(1))
				}
				for i = 3; i <= n; i++ {
					b.Set(i-1, j-1, (b.Get(i-1, j-1)-du.Get(i-1-1)*b.Get(i-1-1, j-1)-du2.Get(i-2-1)*b.Get(i-2-1, j-1))/d.Get(i-1))
				}
				for i = n - 1; i >= 1; i-- {
					if ipiv[i-1] == i {
						b.Set(i-1, j-1, b.Get(i-1, j-1)-dl.Get(i-1)*b.Get(i, j-1))
					} else {
						temp = b.Get(i, j-1)
						b.Set(i, j-1, b.Get(i-1, j-1)-dl.Get(i-1)*temp)
						b.Set(i-1, j-1, temp)
					}
				}
			}
		}
	}
}
