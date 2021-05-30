package eig

import "fmt"

// Alahdg prints header information for the different test paths.
func Alahdg(path []byte) {
	var itype int
	c2 := path[:3]

	//     First line describing matrices in this path
	if string(c2) == "GQR" {
		itype = 1
		fmt.Printf("\n %3s: GQR factorization of general matrices\n", path)
	} else if string(c2) == "GRQ" {
		itype = 2
		fmt.Printf("\n %3s: GRQ factorization of general matrices\n", path)
	} else if string(c2) == "LSE" {
		itype = 3
		fmt.Printf("\n %3s: LSE Problem\n", path)
	} else if string(c2) == "GLM" {
		itype = 4
		fmt.Printf("\n %3s: GLM Problem\n", path)
	} else if string(c2) == "GSV" {
		itype = 5
		fmt.Printf("\n %3s: Generalized Singular Value Decomposition\n", path)
	} else if string(c2) == "CSD" {
		itype = 6
		fmt.Printf("\n %3s: CS Decomposition\n", path)
	}

	//     Matrix types
	fmt.Printf(" %s\n", func() *[]byte { y := []byte("Matrix types: "); return &y }())

	if itype == 1 {
		fmt.Printf("   %2d: A-diagonal matrix  B-upper triangular\n", 1)
		fmt.Printf("   %2d: A-upper triangular B-upper triangular\n", 2)
		fmt.Printf("   %2d: A-lower triangular B-upper triangular\n", 3)
		fmt.Printf("   %2d: Random matrices cond(A)=100, cond(B)=10,\n", 4)
		fmt.Printf("   %2d: Random matrices cond(A)= sqrt( 0.1/EPS ) cond(B)= sqrt( 0.1/EPS )\n", 5)
		fmt.Printf("   %2d: Random matrices cond(A)= 0.1/EPS cond(B)= 0.1/EPS\n", 6)
		fmt.Printf("   %2d: Matrix scaled near underflow limit\n", 7)
		fmt.Printf("   %2d: Matrix scaled near overflow limit\n", 8)
	} else if itype == 2 {
		fmt.Printf("   %2d: A-diagonal matrix  B-lower triangular\n", 1)
		fmt.Printf("   %2d: A-lower triangular B-diagonal triangular\n", 2)
		fmt.Printf("   %2d: A-lower triangular B-upper triangular\n", 3)
		fmt.Printf("   %2d: Random matrices cond(A)=100, cond(B)=10,\n", 4)
		fmt.Printf("   %2d: Random matrices cond(A)= sqrt( 0.1/EPS ) cond(B)= sqrt( 0.1/EPS )\n", 5)
		fmt.Printf("   %2d: Random matrices cond(A)= 0.1/EPS cond(B)= 0.1/EPS\n", 6)
		fmt.Printf("   %2d: Matrix scaled near underflow limit\n", 7)
		fmt.Printf("   %2d: Matrix scaled near overflow limit\n", 8)
	} else if itype == 3 {
		fmt.Printf("   %2d: A-diagonal matrix  B-upper triangular\n", 1)
		fmt.Printf("   %2d: A-upper triangular B-upper triangular\n", 2)
		fmt.Printf("   %2d: A-lower triangular B-upper triangular\n", 3)
		fmt.Printf("   %2d: Random matrices cond(A)=100, cond(B)=10,\n", 4)
		fmt.Printf("   %2d: Random matrices cond(A)=100, cond(B)=10,\n", 5)
		fmt.Printf("   %2d: Random matrices cond(A)=100, cond(B)=10,\n", 6)
		fmt.Printf("   %2d: Random matrices cond(A)=100, cond(B)=10,\n", 7)
		fmt.Printf("   %2d: Random matrices cond(A)=100, cond(B)=10,\n", 8)
	} else if itype == 4 {
		fmt.Printf("   %2d: A-diagonal matrix  B-lower triangular\n", 1)
		fmt.Printf("   %2d: A-lower triangular B-diagonal triangular\n", 2)
		fmt.Printf("   %2d: A-lower triangular B-upper triangular\n", 3)
		fmt.Printf("   %2d: Random matrices cond(A)=100, cond(B)=10,\n", 4)
		fmt.Printf("   %2d: Random matrices cond(A)=100, cond(B)=10,\n", 5)
		fmt.Printf("   %2d: Random matrices cond(A)=100, cond(B)=10,\n", 6)
		fmt.Printf("   %2d: Random matrices cond(A)=100, cond(B)=10,\n", 7)
		fmt.Printf("   %2d: Random matrices cond(A)=100, cond(B)=10,\n", 8)
	} else if itype == 5 {
		fmt.Printf("   %2d: A-diagonal matrix  B-upper triangular\n", 1)
		fmt.Printf("   %2d: A-upper triangular B-upper triangular\n", 2)
		fmt.Printf("   %2d: A-lower triangular B-upper triangular\n", 3)
		fmt.Printf("   %2d: Random matrices cond(A)=100, cond(B)=10,\n", 4)
		fmt.Printf("   %2d: Random matrices cond(A)= sqrt( 0.1/EPS ) cond(B)= sqrt( 0.1/EPS )\n", 5)
		fmt.Printf("   %2d: Random matrices cond(A)= 0.1/EPS cond(B)= 0.1/EPS\n", 6)
		fmt.Printf("   %2d: Random matrices cond(A)= sqrt( 0.1/EPS ) cond(B)=  0.1/EPS \n", 7)
		fmt.Printf("   %2d: Random matrices cond(A)= 0.1/EPS cond(B)=  sqrt( 0.1/EPS )\n", 8)
	} else if itype == 6 {
		fmt.Printf("   %2d: Random orthogonal matrix (Haar measure)\n", 1)
		fmt.Printf("   %2d: Nearly orthogonal matrix with uniformly distributed angles atan2( S, C ) in CS decomposition\n", 2)
		fmt.Printf("   %2d: Random orthogonal matrix with clustered angles atan2( S, C ) in CS decomposition\n", 3)
	}

	//     Tests performed
	fmt.Printf(" %s\n", func() *[]byte { y := []byte("Test ratios: "); return &y }())

	if itype == 1 {
		//        GQR decomposition of rectangular matrices
		fmt.Printf("   %2d: norm( R - Q' * A ) / ( min( N, M )*norm( A )* EPS )\n", 1)
		fmt.Printf("   %2d: norm( T * Z - Q' * B )  / ( min(P,N)*norm(B)* EPS )\n", 2)
		fmt.Printf("   %2d: norm( I - Q'*Q )   / ( N * EPS )\n", 3)
		fmt.Printf("   %2d: norm( I - Z'*Z )   / ( P * EPS )\n", 4)
	} else if itype == 2 {
		//        GRQ decomposition of rectangular matrices
		fmt.Printf("   %2d: norm( R - A * Q' ) / ( min( N,M )*norm(A) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( T * Q - Z' * B )  / ( min( P,N ) * norm(B)*EPS )\n", 2)
		fmt.Printf("   %2d: norm( I - Q'*Q )   / ( N * EPS )\n", 3)
		fmt.Printf("   %2d: norm( I - Z'*Z )   / ( P * EPS )\n", 4)
	} else if itype == 3 {
		//        LSE Problem
		fmt.Printf("   %2d: norm( A*x - c )  / ( norm(A)*norm(x) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( B*x - d )  / ( norm(B)*norm(x) * EPS )\n", 2)
	} else if itype == 4 {
		//        GLM Problem
		fmt.Printf("   %2d: norm( d - A*x - B*y ) / ( (norm(A)+norm(B) )*(norm(x)+norm(y))*EPS )\n", 1)
	} else if itype == 5 {
		//        GSVD
		fmt.Printf("   %2d: norm( U' * A * Q - D1 * R ) / ( min( M, N )*norm( A ) * EPS )\n", 1)
		fmt.Printf("   %2d: norm( V' * B * Q - D2 * R ) / ( min( P, N )*norm( B ) * EPS )\n", 2)
		fmt.Printf("   %2d: norm( I - U'*U )   / ( M * EPS )\n", 3)
		fmt.Printf("   %2d: norm( I - V'*V )   / ( P * EPS )\n", 4)
		fmt.Printf("   %2d: norm( I - Q'*Q )   / ( N * EPS )\n", 5)
	} else if itype == 6 {
		//        CSD
		fmt.Printf("   2-by-2 CSD\n")
		fmt.Printf("   %2d: norm( U1' * X11 * V1 - C ) / ( max(  P,  Q) * max(norm(I-X'*X),EPS) )\n", 1)
		fmt.Printf("   %2d: norm( U1' * X12 * V2-(-S)) / ( max(  P,M-Q) * max(norm(I-X'*X),EPS) )\n", 2)
		fmt.Printf("   %2d: norm( U2' * X21 * V1 - S ) / ( max(M-P,  Q) * max(norm(I-X'*X),EPS) )\n", 3)
		fmt.Printf("   %2d: norm( U2' * X22 * V2 - C ) / ( max(M-P,M-Q) * max(norm(I-X'*X),EPS) )\n", 4)
		fmt.Printf("   %2d: norm( I - U1'*U1 ) / (   P   * EPS )\n", 5)
		fmt.Printf("   %2d: norm( I - U2'*U2 ) / ( (M-P) * EPS )\n", 6)
		fmt.Printf("   %2d: norm( I - V1'*V1 ) / (   Q   * EPS )\n", 7)
		fmt.Printf("   %2d: norm( I - V2'*V2 ) / ( (M-Q) * EPS )\n", 8)
		fmt.Printf("   %2d: principal angle ordering ( 0 or ULP )\n", 9)
		fmt.Printf("   2-by-1 CSD\n")
		fmt.Printf("   %2d: norm( U1' * X11 * V1 - C ) / ( max(  P,  Q) * max(norm(I-X'*X),EPS) )\n", 10)
		fmt.Printf("   %2d: norm( U2' * X21 * V1 - S ) / ( max(  M-P,Q) * max(norm(I-X'*X),EPS) )\n", 11)
		fmt.Printf("   %2d: norm( I - U1'*U1 ) / (   P   * EPS )\n", 12)
		fmt.Printf("   %2d: norm( I - U2'*U2 ) / ( (M-P) * EPS )\n", 13)
		fmt.Printf("   %2d: norm( I - V1'*V1 ) / (   Q   * EPS )\n", 14)
		fmt.Printf("   %2d: principal angle ordering ( 0 or ULP )\n", 15)
	}
}
